import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral, create_pairwise_gaussian

def apply_dense_crf(image_rgb, probs, num_classes=2, num_iterations=5):
    """
    Applique un DenseCRF pour raffiner un masque de segmentation.
    
    Args:
        image_rgb: numpy array de l'image originale, shape (H, W, 3), type uint8 (0-255).
        probs: numpy array des probabilités du masque initial, shape (num_classes, H, W), valeurs [0, 1].
        num_classes: entier, nombre de classes (ex: 2 pour fond/objet).
        num_iterations: entier, nombre d'étapes de raffinement.
        
    Returns:
        Masque raffiné, numpy array de shape (H, W), type int (indices des classes).
    """
    H, W = image_rgb.shape[:2]
    
    # 1. Initialisation du CRF
    d = dcrf.DenseCRF2D(W, H, num_classes)
    
    # 2. Potentiels Unaires (La confiance de votre modèle de base)
    # unary_from_softmax s'attend à un tableau de shape (num_classes, -1)
    U = unary_from_softmax(probs)
    d.setUnaryEnergy(U)
    
    # 3. Potentiels Binaires (Les règles de lissage spatial et de couleur)
    
    # a) Lissage pur (Spatial) : Pénalise les petits trous de prédiction (bruit poivre et sel)
    # sxy : écart-type spatial (taille du voisinage)
    d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
    
    # b) Lissage aligné sur les contours (Bilateral) : Force le masque à s'aligner sur les contrastes RGB
    # sxy : portée spatiale, srgb : sensibilité aux différences de couleurs
    d.addPairwiseBilateral(sxy=(50, 50), srgb=(20, 20, 20), rgbim=image_rgb, 
                           compat=10, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
    
    # 4. Inférence (Message passing)
    Q = d.inference(num_iterations)
    
    # Récupération de la classe la plus probable pour chaque pixel
    refined_mask = np.argmax(Q, axis=0).reshape((H, W))
    
    return refined_mask