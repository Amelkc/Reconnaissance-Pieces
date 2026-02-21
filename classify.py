import numpy as np
import cv2

def get_valeur_piece(label, rayon_px):
    """
    Cette fonction associe une valeur monétaire à une pièce en fonction de son label (type) et de son rayon en pixels.

    Cette fonction mappe les catégories de pièces (cuivre, or, 1 euro, 2 euros) à leurs valeurs correspondante.
    On utilise des seuils de rayons pour distinguer les sous-catégories (ex: 1c vs 2c vs 5c pour les pièces cuivre).

    Args:
        label (str): label de la pièce (ex: 'Cuivre', '1 EURO', etc.).
        rayon_px (int): rayon de la pièce en pixels (sert à différencier les tailles).

    Returns:
        float: Valeur monétaire de la pièce (en euros). Retourne 0.00 si inconnue.
    """

    # Pour les centimes en cuivre
    if 'Cuivre' in label:
        if rayon_px < 50 :
            return 0.01 #1 centime
        elif rayon_px < 70 :
            return 0.02 #2 centimes
        else :
            return 0.05 #5 centimes
        
    # Pour les euros
    elif label == '1 EURO' :
        return 1.00
    elif label == '2 EUROS' :
        return 2.00
    
    #Pour les centimes dorés
    elif 'Or' in label :
        if rayon_px < 60 :
            return 0.10 #10 centimes 
        elif rayon_px < 80 :
            return 0.20 #20 centimes
        else :
            return 0.50 #50 centimes
        
    return 0.00 #Si on ne détecte rien, 0 euros


def classifier_piece(roi_img):
    """
    Cette fonction effectue la classification à partir de l'analyse de couleurs HSV.
    Elle extrait des statistiques de teinte (hue) et de saturation dans le cœur (centre)
    et la couronne (contour) de la pièce, puis applique des règles heuristiques pour déterminer le type (cuivre, or, 1 euro, 2 euros).
    Elle renvoie également une couleur BGR pour l'affichage.

    Args:
        roi_img (numpy.ndarray): région d'intérêt de la pièce (image BGR centrée sur la pièce).

    Returns:
        tuple: (label str, couleur BGR tuple).
    """

    h, w = roi_img.shape[:2]
    if h == 0 or w == 0 :
        return "Inconnu", (128, 128, 128)
    
    cx, cy = w // 2, h // 2
    rayon_max = min(w, h) // 2

    if rayon_max < 5 :
        return "Inconnu", (128, 128, 128)

   # On effectue le prétraitement : conversion en LAB, application de CLAHE sur le canal L
   # et retour en BGR et HSV
    lab = cv2.cvtColor(roi_img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(3,3))
    l = clahe.apply(l)
    lab_norm = cv2.merge([l, a, b])
    roi_norm = cv2.cvtColor(lab_norm, cv2.COLOR_LAB2BGR)
    hsv = cv2.cvtColor(roi_norm, cv2.COLOR_BGR2HSV)

    # On crée les masques pour cœur (centre) et couronne (contour) de la piece
    mask_coeur = np.zeros((h, w), dtype=np.uint8) #coeur central
    cv2.circle(mask_coeur, (cx, cy), int(rayon_max * 0.4), 255, -1)

    mask_full = np.zeros((h, w), dtype=np.uint8) #zone complète
    cv2.circle(mask_full, (cx, cy), int(rayon_max * 0.9), 255, -1)

    mask_inner = np.zeros((h, w), dtype=np.uint8) #zone intermédiare
    cv2.circle(mask_inner, (cx, cy), int(rayon_max * 0.6), 255, -1)

    mask_couronne = cv2.subtract(mask_full, mask_inner) #couronne, donc contour autour du coeur

    # On calcule les moyennes HSV dans le coeur et la couronne
    mean_coeur = cv2.mean(hsv, mask=mask_coeur)
    mean_couronne = cv2.mean(hsv, mask=mask_couronne)
    h_coeur, s_coeur = mean_coeur[0], mean_coeur[1]
    h_cour, s_cour = mean_couronne[0], mean_couronne[1]

    # On classifie en se basant sur la teinte (hue) et la saturation
    is_red_coeur = (h_coeur < 15 or h_coeur > 165) #teinte rouge pour le cuivre
    is_red_cour = (h_cour < 15 or h_cour > 165)

    if (is_red_coeur or is_red_cour) and (s_coeur > 30 or s_cour > 30) :
        return "Cuivre (1,2,5c)", (0, 0, 255)
    
    diff_sat = abs(s_coeur - s_cour) #différence de saturation entre le coeur et la couronne

    if diff_sat > 15 :
        if s_coeur < s_cour :
            return "1 EURO", (255, 0, 0) #couronne saturée
        
        elif 15 < h_coeur < 40 :
            return "2 EUROS", (0, 255, 0) #teinte jaune-orange

    return "Or (10,20,50c)", (0, 255, 255) #saturation uniforme






