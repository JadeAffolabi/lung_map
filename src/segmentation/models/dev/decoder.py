"""
models/decoder.py — Décodeur UNet amélioré
===========================================

Améliorations par rapport à la version initiale :

  1. ResBlock avec SE (Squeeze-Excitation)
     Remplace le DoubleConv. La connexion résiduelle stabilise le gradient.
     Le bloc SE recalibre les canaux : quels filtres activer dans ce contexte.

  2. ASPP (Atrous Spatial Pyramid Pooling) au bottleneck
     Agrège du contexte à plusieurs échelles (dilatations 1,6,12,18) sans
     réduire la résolution. Compensé le fait que les features ViT à H/32
     ont un champ récepteur global mais perdent la structure locale fine.

  3. FPN (Feature Pyramid Network) comme neck
     Fusion top-down avant le décodeur UNet : chaque niveau reçoit une
     contribution du niveau plus profond. Améliore la cohérence multi-échelle.

  4. Attention Gates (inchangés dans le principe, intégrés dans les UpBlocks)
     Filtrent les skip connections pour ne garder que les régions pertinentes.

  5. Deep supervision
     Têtes auxiliaires aux niveaux intermédiaires — stabilise l'entraînement
     en début de convergence, surtout avec un backbone gelé ou LoRA faible rank.

Flux de données :
  [f0,f1,f2,f3] (depuis FoundationEncoder)
       │
    FPN Neck      → [p0,p1,p2,p3]  (fusion top-down, même dim proj_dim)
       │
    ASPP          → contexte enrichi sur p3 (bottleneck)
       │
  UpBlock x3     → + skip connections (p2, p1, p0) avec Attention Gates
       │
  Seg Head       → logits (B, num_classes, H, W)

🔬 EXPÉRIMENTATION : voir annotations [EXP] dans chaque classe.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict
from config import DecoderConfig


# ═══════════════════════════════════════════════════════════════════════════════
# 1. UTILITAIRES
# ═══════════════════════════════════════════════════════════════════════════════

def get_norm(norm_type: str, num_channels: int) -> nn.Module:
    """
    🔬 [EXP]
      'bn' : BatchNorm2d   → stable, batchs ≥ 4
      'gn' : GroupNorm(32) → batchs petits ou features ViT (recommandé ici)
      'in' : InstanceNorm  → standard imagerie médicale
    """
    if norm_type == "bn":
        return nn.BatchNorm2d(num_channels)
    elif norm_type == "gn":
        num_groups = min(32, num_channels)
        return nn.GroupNorm(num_groups, num_channels)
    elif norm_type == "ln":
        return nn.LayerNorm(num_channels)
    elif norm_type == "in":
        return nn.InstanceNorm2d(num_channels, affine=True)
    raise ValueError(f"Normalisation inconnue : {norm_type}")


def get_act(act_type: str) -> nn.Module:
    """
    🔬 [EXP] GELU cohérent avec le ViT encodeur vs ReLU classique UNet.
    """
    mapping = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}
    return mapping[act_type]()


def conv1x1(in_c: int, out_c: int) -> nn.Conv2d:
    return nn.Conv2d(in_c, out_c, kernel_size=1, bias=False)


def conv3x3(in_c: int, out_c: int, dilation: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_c, out_c, kernel_size=3,
        padding=dilation, dilation=dilation, bias=False
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 2. SQUEEZE-EXCITATION
# ═══════════════════════════════════════════════════════════════════════════════

class SEBlock(nn.Module):
    """
    Recalibre les canaux par attention globale : quels filtres sont utiles ici ?

      x → AvgPool → FC → ReLU → FC → Sigmoid → scale(x)

    🔬 [EXP]
      - reduction : 16 (standard) vs 8 (plus expressif) vs 32 (plus léger)
      - Remplacer par CBAM (ajoute une attention spatiale en plus)
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, _, _ = x.shape
        scale = self.pool(x).view(B, C)
        scale = self.fc(scale).view(B, C, 1, 1)
        return x * scale


# ═══════════════════════════════════════════════════════════════════════════════
# 3. RESIDUAL BLOCK AVEC SE
# ═══════════════════════════════════════════════════════════════════════════════

class ResBlock(nn.Module):
    """
    Bloc résiduel : Conv-Norm-Act → Dropout → Conv-Norm → SE → + skip → Act

    Avantages vs DoubleConv :
      • Connexion résiduelle : gradient circule directement, convergence plus stable
      • SE : recalibre les canaux sans coût mémoire important
      • Compatible avec tout ratio in_c / out_c (projection 1x1 si nécessaire)

    🔬 [EXP]
      - Ajouter une 3ème conv (bottleneck ResNet-style) pour les grands channels
      - Essayer Depthwise-Separable Conv pour alléger le décodeur
      - se_reduction : 16 (défaut) vs 8 vs 32
    """

    def __init__(
        self,
        in_c: int,
        out_c: int,
        norm: str = "gn",
        act: str = "gelu",
        dropout: float = 0.0,
        se_reduction: int = 16,
    ):
        super().__init__()
        self.conv1 = nn.Sequential(
            conv3x3(in_c, out_c),
            get_norm(norm, out_c),
            get_act(act),
        )
        self.drop = nn.Dropout2d(dropout)
        self.conv2 = nn.Sequential(
            conv3x3(out_c, out_c),
            get_norm(norm, out_c),
        )
        self.se = SEBlock(out_c, reduction=se_reduction)
        self.act = get_act(act)

        # Projection résiduelle si les dimensions changent
        self.shortcut = (
            nn.Sequential(conv1x1(in_c, out_c), get_norm(norm, out_c))
            if in_c != out_c else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.drop(out)
        out = self.conv2(out)
        out = self.se(out)
        return self.act(out + residual)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. ASPP — ATROUS SPATIAL PYRAMID POOLING
# ═══════════════════════════════════════════════════════════════════════════════

class ASPP(nn.Module):
    """
    Capture du contexte multi-échelle sur la feature map la plus profonde
    sans réduire la résolution spatiale.

    Branches :
      • Conv 1x1           (contexte local)
      • Conv 3x3 dilatée x6, x12, x18  (contexte moyen à large)
      • Global Average Pooling          (contexte global)
    → Concat + Conv 1x1 de fusion

    Pourquoi ici ?
    Les features ViT à H/32 ont déjà un champ récepteur global (attention),
    mais ASPP ajoute un biais inductif *local et multi-échelle* qui aide le
    décodeur CNN à affiner les bordures et les petites structures.

    🔬 [EXP]
      - dilations : (6,12,18) standard DeepLabv3 vs (3,6,12) pour H/32 petite
      - out_channels : 256 (défaut) vs 512 si proj_dim élevé
      - Désactiver ASPP si la feature map H/32 est déjà trop petite (<7x7)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dilations: tuple = (6, 12, 18),
        norm: str = "gn",
        act: str = "gelu",
    ):
        super().__init__()

        def _branch(dilation: int) -> nn.Sequential:
            k = 1 if dilation == 1 else 3
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, k,
                          padding=0 if k == 1 else dilation,
                          dilation=dilation, bias=False),
                get_norm(norm, out_channels),
                get_act(act),
            )

        self.b1 = _branch(1)
        self.b2 = _branch(dilations[0])
        self.b3 = _branch(dilations[1])
        self.b4 = _branch(dilations[2])

        # Branche pooling global
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            conv1x1(in_channels, out_channels),
            get_norm(norm, out_channels),
            get_act(act),
        )

        # Fusion des 5 branches
        self.fuse = nn.Sequential(
            conv1x1(out_channels * 5, out_channels),
            get_norm(norm, out_channels),
            get_act(act),
            nn.Dropout2d(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[2:]
        gap = F.interpolate(self.gap(x), size=(h, w), mode="bilinear", align_corners=False)
        out = torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x), gap], dim=1)
        return self.fuse(out)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. FPN NECK
# ═══════════════════════════════════════════════════════════════════════════════

class FPNNeck(nn.Module):
    """
    Aligne et fusionne les 4 niveaux de l'encodeur (top-down).

    Toutes les features entrantes ont déjà `proj_dim` canaux (depuis l'encodeur).
    Le FPN ajoute une fusion top-down : chaque niveau reçoit l'information
    des niveaux plus profonds via un upsampling + addition.

    Pourquoi ici et pas dans le décodeur UNet ?
    Le FPN opère *avant* les skip connections UNet. Il enrichit chaque niveau
    avec un contexte global *avant* que le décodeur ne les fusionne avec
    les features upsamplées. Les skip connections UNet n'ont alors plus
    besoin de "combler" un manque d'information globale.

    🔬 [EXP]
      - Désactiver le FPN et comparer (use_fpn=False dans DecoderConfig)
      - Remplacer l'addition par une concat + conv1x1 (plus de paramètres)
      - Ajouter un BiFPN (pondération apprise des fusions)
    """

    def __init__(self, channels: int, norm: str = "gn", act: str = "gelu"):
        super().__init__()
        # Conv 3x3 post-fusion pour lisser les artefacts d'upsampling
        self.smooth = nn.ModuleList([
            nn.Sequential(
                conv3x3(channels, channels),
                get_norm(norm, channels),
                get_act(act),
            )
            for _ in range(4)
        ])

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        features = [f0 (grande), f1, f2, f3 (petite)]
        Fusion top-down : f3 → f2 → f1 → f0
        """
        # Copie pour ne pas modifier les tenseurs originaux
        p = list(features)

        for i in range(len(p) - 2, -1, -1):          # 2, 1, 0
            upsampled = F.interpolate(
                p[i + 1], size=p[i].shape[2:],
                mode="bilinear", align_corners=False
            )
            p[i] = p[i] + upsampled

        return [smooth(pi) for smooth, pi in zip(self.smooth, p)]


# ═══════════════════════════════════════════════════════════════════════════════
# 6. ATTENTION GATE
# ═══════════════════════════════════════════════════════════════════════════════

class AttentionGate(nn.Module):
    """
    Filtre la skip connection par une carte d'attention apprise.

    Ref : Oktay et al., "Attention U-Net", 2018

      g   : signal « guide » montant du décodeur (sémantique riche, basse résolution)
      x   : skip connection (détails fins, haute résolution)
      → alpha = sigmoid( W_ψ · relu(W_g(g) + W_x(x)) )
      → sortie = x · alpha

    🔬 [EXP]
      - Remplacer par CBAM (Convolutional Block Attention Module)
        qui combine attention canal + spatiale
      - Désactiver pour mesurer son apport réel (use_attention_gate=False)
    """

    def __init__(self, g_channels: int, x_channels: int, inter_channels: int):
        super().__init__()
        self.W_g = nn.Sequential(
            conv1x1(g_channels, inter_channels),
            nn.BatchNorm2d(inter_channels),
        )
        self.W_x = nn.Sequential(
            conv1x1(x_channels, inter_channels),
            nn.BatchNorm2d(inter_channels),
        )
        self.psi = nn.Sequential(
            conv1x1(inter_channels, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU()

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # Aligner g sur la résolution de x
        g_up = F.interpolate(self.W_g(g), size=x.shape[2:],
                             mode="bilinear", align_corners=False)
        alpha = self.psi(self.relu(g_up + self.W_x(x)))
        return x * alpha


# ═══════════════════════════════════════════════════════════════════════════════
# 7. UPBLOCK
# ═══════════════════════════════════════════════════════════════════════════════

class UpBlock(nn.Module):
    """
    Étape de décodage :
      1. Bilinear x2
      2. Attention Gate sur le skip FPN (optionnel)
      3. Concat [upsampled, skip]
      4. ResBlock avec SE

    🔬 [EXP]
      - Remplacer bilinear par ConvTranspose2d (appris, peut créer des artefacts)
      - Remplacer bilinear par PixelShuffle (sub-pixel convolution)
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        use_attention: bool = True,
        norm: str = "gn",
        act: str = "gelu",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.use_attention = use_attention and skip_channels > 0

        if self.use_attention:
            self.attn = AttentionGate(
                g_channels=in_channels,
                x_channels=skip_channels,
                inter_channels=max(out_channels // 2, 1),
            )

        self.res = ResBlock(
            in_c=in_channels + skip_channels,
            out_c=out_channels,
            norm=norm,
            act=act,
            dropout=dropout,
        )

    def forward(
        self, x: torch.Tensor, skip: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

        if skip is not None:
            if self.use_attention:
                skip = self.attn(x, skip)
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:],
                                  mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)

        return self.res(x)


# ═══════════════════════════════════════════════════════════════════════════════
# 8. DÉCODEUR COMPLET
# ═══════════════════════════════════════════════════════════════════════════════

class UNetDecoder(nn.Module):
    """
    Décodeur UNet amélioré.

    Entrée  : [f0 (H/8), f1 (H/16), f2 (H/32), f3 (H/64)]  — proj_dim canaux
    Sortie  : {"logits": (B, C, H, W)} + auxiliaires si deep_supervision

    Pipeline :
      FPNNeck → ASPP(f3) → UpBlockx3 → UpBlock final → SegHead

    Résolutions (patch_size=16, img=224, out_indices=[5,11,17,23]) :
      f0 : 28x28 (H/8)   f1 : 14x14 (H/16)
      f2 : 7x7  (H/32)   f3 : 4x4   (H/64 approx.)

    Résolutions (patch_size=16, img=512, out_indices=[5,11,17,23]) :
      f0 : 64x64          f1 : 32x32
      f2 : 16x16          f3 : 8x8

    🔬 [EXP] :
      channels   : [512,256,128,64] (large) vs [256,128,64,32] (léger)
      use_fpn    : True/False — mesurer l'apport du neck
      aspp_dil   : (6,12,18) standard vs (3,6,12) si feature map H/32 < 10x10
      norm_type  : "gn" (recommandé avec peft/petit batch) vs "bn"
    """

    def __init__(self, cfg: DecoderConfig, encoder_dim: int):
        super().__init__()
        ch = cfg.channels            # [ch0, ch1, ch2, ch3]
        enc = encoder_dim            # proj_dim de l'encodeur
        norm = cfg.norm_type
        act = cfg.activation
        drop = cfg.dropout_rate
        use_attn = cfg.use_attention_gate
        self.deep_supervision = cfg.deep_supervision

        # ── FPN Neck ───────────────────────────────────────────────────────
        self.use_fpn = getattr(cfg, "use_fpn", True)
        if self.use_fpn:
            self.fpn = FPNNeck(channels=enc, norm=norm, act=act)

        # ── Bottleneck ASPP ────────────────────────────────────────────────
        self.aspp = ASPP(
            in_channels=enc,
            out_channels=ch[0],
            dilations=getattr(cfg, "aspp_dilations", (6, 12, 18)),
            norm=norm, act=act,
        )

        # ── Blocs de décodage ──────────────────────────────────────────────
        # up1 : H/32 → H/16  (skip = f2 après FPN)
        self.up1 = UpBlock(ch[0], enc, ch[1], use_attn, norm, act, drop)
        # up2 : H/16 → H/8   (skip = f1 après FPN)
        self.up2 = UpBlock(ch[1], enc, ch[2], use_attn, norm, act, drop)
        # up3 : H/8  → H/4   (skip = f0 après FPN)
        self.up3 = UpBlock(ch[2], enc, ch[3], use_attn, norm, act, drop)
        # up4 : H/4  → H/2   (pas de skip — on quitte la zone de l'encodeur)
        self.up4 = UpBlock(ch[3], 0,   ch[3], False,    norm, act, drop)

        # ── Tête de segmentation ───────────────────────────────────────────
        self.seg_head = nn.Sequential(
            ResBlock(ch[3], ch[3], norm=norm, act=act),
            conv1x1(ch[3], cfg.num_classes),
        )

        # ── Supervision profonde ───────────────────────────────────────────
        if self.deep_supervision:
            # Têtes légères (conv 1x1) aux sorties intermédiaires
            self.aux1 = conv1x1(ch[1], cfg.num_classes)   # après up1 (H/16)
            self.aux2 = conv1x1(ch[2], cfg.num_classes)   # après up2 (H/8)
            self.aux3 = conv1x1(ch[3], cfg.num_classes)   # après up3 (H/4)

    def forward(
        self,
        features: List[torch.Tensor],
        target_size: Optional[tuple] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        features : [f0 (grande), f1, f2, f3 (petite)]
        target_size : résolution de sortie finale (H_img, W_img)
        """
        target_size = target_size or features[0].shape[2:]

        # ── FPN ─────────────────────────────────────────────────────────
        f0, f1, f2, f3 = self.fpn(features) if self.use_fpn else features

        # ── Bottleneck ASPP ──────────────────────────────────────────────
        x = self.aspp(f3)          # (B, ch[0], H/64, W/64)

        # ── Décodage ────────────────────────────────────────────────────
        x = self.up1(x, f2)        # (B, ch[1], H/32, W/32)
        a1 = x

        x = self.up2(x, f1)        # (B, ch[2], H/16, W/16)
        a2 = x

        x = self.up3(x, f0)        # (B, ch[3], H/8,  W/8)
        a3 = x

        x = self.up4(x)            # (B, ch[3], H/4,  W/4)

        # ── Upsampling final ─────────────────────────────────────────────
        x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        logits = self.seg_head(x)

        output = {"logits": logits}

        if self.deep_supervision and self.training:
            def aux_up(t, head):
                return head(F.interpolate(t, size=target_size,
                                          mode="bilinear", align_corners=False))
            output["aux1"] = aux_up(a1, self.aux1)
            output["aux2"] = aux_up(a2, self.aux2)
            output["aux3"] = aux_up(a3, self.aux3)

        return output















class SimpleFPN(nn.Module):
    """
    🔬 [EXP] :
      - out_channels : [64,128,256,512] standard vs [128,256,512,512] plus riche
      - norm_type    : "gn" recommandé (cohérent avec le décodeur)
      - Ajouter une conv 3×3 supplémentaire après chaque upsampling
    """

    def __init__(
        self,
        embed_dim: int,
        out_channels: List[int],    # [c_H4, c_H8, c_H16, c_H32]
        norm: str = "gn",
        act: str = "gelu",
    ):
        super().__init__()
        c0, c1, c2, c3 = out_channels  # du plus grand au plus petit spatial

        def up_block(in_c, out_c):
            """Bilinear ×2 + Conv3×3 — évite les artefacts ConvTranspose2d."""
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                conv3x3(in_c, out_c),
                get_norm(norm, out_c),
                get_act(act),
            )

        # H/4 : ×4 depuis H/16 (deux upsampling successifs)
        self.scale_1_4 = nn.Sequential(
            up_block(embed_dim, c0),
            up_block(c0, c0),           # second ×2 avec les canaux déjà réduits
        )

        # H/8 : ×2 depuis H/16
        self.scale_1_8 = up_block(embed_dim, c1)

        # H/16 : même résolution, projection 1×1
        self.scale_1_16 = nn.Sequential(
            conv1x1(embed_dim, c2),
            get_norm(norm, c2),
            get_act(act),
        )

        # H/32 : ×2 downsampling depuis H/16
        self.scale_1_32 = nn.Sequential(
            nn.Conv2d(embed_dim, c3, kernel_size=3, stride=2, padding=1, bias=False),
            get_norm(norm, c3),
            get_act(act),
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        x : (B, embed_dim, H/16, W/16)
        → [f0 (H/4), f1 (H/8), f2 (H/16), f3 (H/32)]
        """
        return [
            self.scale_1_4(x),    # (B, c0, H/4,  W/4)
            self.scale_1_8(x),    # (B, c1, H/8,  W/8)
            self.scale_1_16(x),   # (B, c2, H/16, W/16)
            self.scale_1_32(x),   # (B, c3, H/32, W/32)
        ]


class UNetDecoder2(nn.Module):


    def __init__(self, cfg: DecoderConfig, encoder_channels: List[int]):
        super().__init__()
        ch = cfg.channels            # [ch0, ch1, ch2, ch3]
        norm = cfg.norm_type
        act = cfg.activation
        drop = cfg.dropout_rate
        use_attn = cfg.use_attention_gate
        self.deep_supervision = cfg.deep_supervision

        c_f0, c_f1, c_f2, c_f3 = encoder_channels


        # ── Bottleneck ASPP ────────────────────────────────────────────────
        self.aspp = ASPP(
            in_channels=c_f3,
            out_channels=ch[0],
            dilations=getattr(cfg, "aspp_dilations", (6, 12, 18)),
            norm=norm, act=act,
        )

        # ── Blocs de décodage ──────────────────────────────────────────────
        self.up1 = UpBlock(ch[0], c_f2, ch[1], use_attn, norm, act, drop)
        self.up2 = UpBlock(ch[1], c_f1, ch[2], use_attn, norm, act, drop)
        self.up3 = UpBlock(ch[2], c_f0, ch[3], use_attn, norm, act, drop)

        # ── Tête de segmentation ───────────────────────────────────────────
        self.seg_head = nn.Sequential(
            ResBlock(ch[3], ch[3], norm=norm, act=act),
            conv1x1(ch[3], cfg.num_classes),
        )

        # ── Supervision profonde ───────────────────────────────────────────
        if self.deep_supervision:
            # Têtes légères (conv 1x1) aux sorties intermédiaires
            self.aux1 = conv1x1(ch[1], cfg.num_classes)   # après up1 (H/16)
            self.aux2 = conv1x1(ch[2], cfg.num_classes)   # après up2 (H/8)
            self.aux3 = conv1x1(ch[3], cfg.num_classes)   # après up3 (H/4)

    def forward(
        self,
        features: List[torch.Tensor],
        target_size: Optional[tuple] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        features : [f0 (grande), f1, f2, f3 (petite)]
        target_size : résolution de sortie finale (H_img, W_img)
        """

        f0, f1, f2, f3 = features
        target_size = target_size

        # ── FPN ─────────────────────────────────────────────────────────

        # ── Bottleneck ASPP ──────────────────────────────────────────────
        x = self.aspp(f3)          # (B, ch[0], H/64, W/64)

        # ── Décodage ────────────────────────────────────────────────────
        x = self.up1(x, f2)        # (B, ch[1], H/32, W/32)
        a1 = x

        x = self.up2(x, f1)        # (B, ch[2], H/16, W/16)
        a2 = x

        x = self.up3(x, f0)        # (B, ch[3], H/8,  W/8)
        a3 = x


        # ── Upsampling final ─────────────────────────────────────────────
        x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        logits = self.seg_head(x)

        output = {"logits": logits}

        if self.deep_supervision and self.training:
            def aux_up(t, head):
                return head(F.interpolate(t, size=target_size,
                                          mode="bilinear", align_corners=False))
            output["aux1"] = aux_up(a1, self.aux1)
            output["aux2"] = aux_up(a2, self.aux2)
            output["aux3"] = aux_up(a3, self.aux3)

        return output