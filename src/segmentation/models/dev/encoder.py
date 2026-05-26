"""
models/encoder.py — Encodeur Foundation Model (UNI / CONCH) + LoRA via peft
=============================================================================

Chargement :
  UNI   → timm  (hf-hub:MahmoodLab/UNI)  — ViT-Large patch16
  CONCH → HuggingFace AutoModel           — ViT-Base/Large

Fine-tuning :
  LoRA via peft.get_peft_model → gèle le backbone, n'entraîne que A et B
  Bénéfices vs implémentation manuelle :
    • merge_and_unload()       → fusionne A·B dans W, zéro overhead à l'inférence
    • save_pretrained()        → sauvegarde uniquement les deltas (~Mo, pas Go)
    • Variantes gratuites      → DoRA, QLoRA, LoHA selon peft version
    • print_trainable_params() → diagnostic immédiat du % de paramètres actifs

Extraction multi-échelle :
  Forward hooks sur les blocs ViT aux indices `out_indices`.
  Compatible avec n'importe quel wrapper (peft, DataParallel, etc.)
  Chaque feature map extraite est projetée vers `proj_dim` canaux uniformes.

🔬 EXPÉRIMENTATION :
  - lora_rank          : 4 → 64 (capacité vs régularisation)
  - lora_alpha         : scaling = alpha/rank, typiquement ratio ∈ [1, 2]
  - lora_target_modules: ["qkv"] seul vs ["qkv","proj"] vs +["fc1","fc2"]
  - out_indices        : [5,11,17,23] équidistants vs [3,7,15,23] focus profond
  - proj_dim           : 256 (léger) vs 512 (riche), à aligner avec le décodeur
"""

import timm
import torch
import torch.nn as nn
from typing import List, Dict, Optional
from peft import LoraConfig, get_peft_model
from config import EncoderConfig


# ═══════════════════════════════════════════════════════════════════════════════
# Chargement des backbones
# ═══════════════════════════════════════════════════════════════════════════════

def load_uni(cfg: EncoderConfig) -> nn.Module:
    """
    UNI : ViT-Large pré-entraîné sur 100M+ patches de WSIs (MahmoodLab).
    Chargé via timm depuis le hub HuggingFace.

    Pré-requis :
      pip install timm huggingface_hub
      huggingface-cli login   # accès restreint, demander sur HF

    Ref : Chen et al., "A General-Purpose Self-Supervised Model for Computational Pathology", 2024
    """
    model = timm.create_model(
        "hf-hub:MahmoodLab/UNI",
        pretrained=True,
        init_values=1e-5,       # LayerScale — valeur du papier UNI
        dynamic_img_size=True,  # Accepte des tailles > 224 au test
    )
    model.head = nn.Identity()  # Supprime la tête de classification
    return model


def load_conch(cfg: EncoderConfig):
    """
    CONCH : ViT pré-entraîné avec supervision contrastive image-texte (MahmoodLab).
    On extrait uniquement l'encodeur vision.

    Pré-requis :
      pip install transformers huggingface_hub
      huggingface-cli login

    Ref : Lu et al., "A visual-language foundation model for computational pathology", 2024
    """

    """ # AutoModel retourne le modèle complet (vision + text) ;
    # on isole l'encodeur vision via l'attribut exposé par le custom code
    full_model = AutoModel.from_pretrained(
        "MahmoodLab/CONCH",
        trust_remote_code=True,
    )
    # L'encodeur vision est accessible via .vision_model ou .visual selon la version
    vision_encoder = getattr(
        full_model, "vision_model", getattr(full_model, "visual", full_model)
    )
    return vision_encoder """
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# Extraction multi-échelle via hooks
# ═══════════════════════════════════════════════════════════════════════════════

class IntermediateFeatureExtractor:
    """
    Enregistre des forward hooks sur les blocs ViT spécifiés.
    Fonctionne avec n'importe quel wrapper autour du backbone (peft, DDP…).

    Usage :
        extractor = IntermediateFeatureExtractor(backbone, out_indices=[5,11,17,23])
        _ = backbone(x)
        features = extractor.get_features()   # dict {idx: tensor}
        extractor.clear()
    """

    def __init__(self, model: nn.Module, out_indices: List[int]):
        self._features: Dict[int, torch.Tensor] = {}
        self._hooks = []

        # Récupère les blocs ViT, qu'ils soient enveloppés par peft ou non
        blocks = self._find_blocks(model)

        for idx in out_indices:
            hook = blocks[idx].register_forward_hook(
                lambda _, __, output, i=idx: self._features.__setitem__(i, output)
            )
            self._hooks.append(hook)

    @staticmethod
    def _find_blocks(model: nn.Module) -> nn.ModuleList:
        """
        Cherche l'attribut `blocks` dans le modèle ou son backbone sous-jacent.
        Robuste aux wrappers peft (PeftModel enveloppe dans model.base_model.model…).
        """
        for attr in ["blocks", "base_model.model.blocks", "model.blocks",
                     "base_model.blocks", "vision_model.encoder.layers"]:
            obj = model
            found = True
            for part in attr.split("."):
                if hasattr(obj, part):
                    obj = getattr(obj, part)
                else:
                    found = False
                    break
            if found and isinstance(obj, (nn.ModuleList, nn.Sequential)):
                return obj
        raise AttributeError(
            "Impossible de localiser les blocs ViT. "
            "Vérifier l'attribut 'blocks' dans votre backbone."
        )

    def get_features(self) -> Dict[int, torch.Tensor]:
        return dict(self._features)

    def clear(self):
        self._features.clear()

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()


# ═══════════════════════════════════════════════════════════════════════════════
# Encodeur principal
# ═══════════════════════════════════════════════════════════════════════════════

class FoundationEncoder(nn.Module):
    """
    Encodeur Foundation Model avec LoRA (peft) et extraction multi-échelle.

    Sortie : liste de 4 feature maps projetées vers proj_dim canaux
      [f0 (H/8, W/8), f1 (H/16), f2 (H/32), f3 (H/64)]
      (les résolutions exactes dépendent du patch_size=16 et de out_indices)

    Sauvegarde recommandée :
      # Uniquement les poids LoRA (quelques Mo)
      model.encoder.backbone.save_pretrained("checkpoints/lora/")

      # Rechargement
      from peft import PeftModel
      base = load_uni(cfg)
      backbone = PeftModel.from_pretrained(base, "checkpoints/lora/")

      # Fusion pour inférence sans overhead
      backbone = backbone.merge_and_unload()
    """

    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.out_indices = cfg.out_indices

        # ── 1. Chargement du backbone ──────────────────────────────────────
        loaders = {"uni": load_uni, "conch": load_conch}
        if cfg.name not in loaders:
            raise ValueError(f"Backbone inconnu : {cfg.name}. Choisir parmi {list(loaders)}")
        self.backbone = loaders[cfg.name](cfg)

        # ── 2. Application de LoRA via peft ───────────────────────────────
        if cfg.finetune_strategy == "lora":
            lora_config = LoraConfig(
                r=cfg.lora_rank,                         # 🔬 [EXP] 4,8,16,32,64
                lora_alpha=cfg.lora_alpha,               # 🔬 [EXP] scaling = alpha/r
                lora_dropout=cfg.lora_dropout,           # 🔬 [EXP] 0.0, 0.05, 0.1
                target_modules=cfg.lora_target_modules,  # 🔬 [EXP] voir docstring module
                bias="none",
                # Pas de task_type : on adapte un encodeur, pas un LLM
            )
            self.backbone = get_peft_model(self.backbone, lora_config)
            self.backbone.print_trainable_parameters()

        elif cfg.finetune_strategy == "frozen":
            for p in self.backbone.parameters():
                p.requires_grad = False

        elif cfg.finetune_strategy == "full":
            pass  # Tout entraînable — nécessite lr_encoder très faible (~1e-6)

        else:
            raise ValueError(f"Stratégie inconnue : {cfg.finetune_strategy}")

        # ── 3. Hooks d'extraction multi-échelle ───────────────────────────
        self._extractor = IntermediateFeatureExtractor(self.backbone, self.out_indices)

        # ── 4. Projections 1×1 : embed_dim → proj_dim (uniforme) ──────────
        # On ne connaît pas embed_dim statiquement pour CONCH vs UNI,
        # on le détecte au premier forward (lazy init).
        self.proj_dim = cfg.proj_dim
        self._proj_initialized = False
        self.feature_projs: Optional[nn.ModuleList] = None

    def _init_projections(self, sample_features: Dict[int, torch.Tensor]):
        """
        Initialise les projections 1×1 après le premier forward,
        une fois la dimension embed_dim connue.
        """
        projs = []
        for idx in self.out_indices:
            feat = sample_features[idx]           # (B, N+1, D) ou (B, D, h, w)
            # Les sorties de blocs ViT timm sont (B, N, D) — séquence de tokens
            embed_dim = feat.shape[-1] if feat.dim() == 3 else feat.shape[1]
            projs.append(nn.Sequential(
                nn.Conv2d(embed_dim, self.proj_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(self.proj_dim),
                nn.GELU(),
            ))
        self.feature_projs = nn.ModuleList(projs).to(
            next(self.backbone.parameters()).device
        )
        self._proj_initialized = True

    @staticmethod
    def _tokens_to_map(tokens: torch.Tensor, has_cls: bool = True) -> torch.Tensor:
        """
        Convertit une sortie de bloc ViT (B, N+1, D) en feature map (B, D, h, w).
        h = w = sqrt(N)  (supposant un grid carré de patches).
        """
        if tokens.dim() == 4:
            return tokens   # Déjà une feature map (certains modèles HF)
        seq = tokens[:, 1:] if has_cls else tokens   # retire le CLS token
        B, N, D = seq.shape
        h = w = int(N ** 0.5)
        return seq.transpose(1, 2).reshape(B, D, h, w)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Retourne 4 feature maps projetées :
          [f0 (H/8), f1 (H/16), f2 (H/32), f3 (H/64)]
          (strides relatifs au patch_size=16 et à out_indices)
        """
        self._extractor.clear()

        # Forward complet du backbone (les hooks capturent les sorties intermédiaires)
        _ = self.backbone(x)
        raw: Dict[int, torch.Tensor] = self._extractor.get_features()

        # Init lazy des projections
        if not self._proj_initialized:
            self._init_projections(raw)

        # Conversion tokens → feature maps + projection
        features = []
        for proj, idx in zip(self.feature_projs, self.out_indices):
            feat_map = self._tokens_to_map(raw[idx])   # (B, D, h, w)
            features.append(proj(feat_map))            # (B, proj_dim, h, w)

        return features   # [f0, f1, f2, f3], résolutions décroissantes
