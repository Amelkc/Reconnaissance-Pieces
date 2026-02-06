import cv2
import numpy as np

def get_hsv_stats(img_hsv, mask):
    """Calcule la teinte et saturation moyennes dans une zone masquée"""
    mean_val = cv2.mean(img_hsv, mask=mask)
    return mean_val[0], mean_val[1] # Hue, Saturation

def is_gold(hue, sat):
    # Jaune/Or : Teinte ~15-35, Saturation modérée à forte
    return (15 <= hue <= 40) and (sat > 25)

def is_copper(hue, sat):
    # Cuivre : Teinte très basse (<15) ou très haute (>165), Saturation forte/très forte
    # "très foncée" = souvent saturation forte mais valeur (V) faible, 
    # mais en HSV la teinte reste rouge.
    return (hue < 15 or hue > 165) and (sat > 40)

def is_silver(hue, sat):
    # Argent : Saturation faible (grisaille), peu importe la teinte
    return sat < 25

def classifier_expert(roi_img):
    """
    Classification robuste aux reflets et à l'oxydation.
    Combine l'analyse structurelle (Coeur/Couronne) ET spectrale (Teinte globale).
    """
    h, w = roi_img.shape[:2]
    cx, cy = w // 2, h // 2
    rayon_max = w // 2
    
    # 1. Conversion & Masques
    hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
    
    # Masque COEUR (40%)
    mask_coeur = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask_coeur, (cx, cy), int(rayon_max * 0.4), 255, -1)
    
    # Masque COURONNE (60-90%)
    mask_full = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask_full, (cx, cy), int(rayon_max * 0.9), 255, -1)
    mask_inner = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask_inner, (cx, cy), int(rayon_max * 0.6), 255, -1)
    mask_couronne = cv2.subtract(mask_full, mask_inner)
    
    # 2. Mesures
    mean_coeur = cv2.mean(hsv, mask=mask_coeur)
    mean_couronne = cv2.mean(hsv, mask=mask_couronne)
    
    h_coeur, s_coeur = mean_coeur[0], mean_coeur[1]
    h_cour, s_cour = mean_couronne[0], mean_couronne[1]
    
    # --- 3. LOGIQUE DE DÉCISION BLINDÉE ---
    
    # RÈGLE 1 (LA SÉCURITÉ) : Est-ce du CUIVRE évident ?
    # Si le centre OU le bord est rouge profond (<15 ou >165), c'est du cuivre.
    # Peu importe les différences de luminosité/saturation dues à la saleté.
    is_red_coeur = (h_coeur < 15 or h_coeur > 165)
    is_red_cour = (h_cour < 15 or h_cour > 165)
    
    if (is_red_coeur or is_red_cour) and (s_coeur > 30 or s_cour > 30):
        return "Cuivre (1,2,5c)", (0, 0, 255)

    # RÈGLE 2 : Les Bicolores (Seulement si ce n'est PAS rouge)
    diff_sat = abs(s_coeur - s_cour)
    
    # On exige une différence nette de saturation (>15)
    if diff_sat > 15:
        # Centre Gris (faible sat) + Bord Coloré (forte sat) -> 1 EURO
        if s_coeur < s_cour:
             return "1 EURO", (255, 0, 0)
        
        # Centre Coloré + Bord Gris -> 2 EUROS
        # AJOUT DE SÉCURITÉ : Le centre doit être JAUNE (pas rouge !)
        elif 15 < h_coeur < 40: 
             return "2 EUROS", (0, 255, 0)

    # RÈGLE 3 : Le reste est de l'Or (10, 20, 50 cts)
    return "Or (10,20,50c)", (0, 255, 255)

def detecter_et_identifier(chemin_image):
    # --- 1. CHARGEMENT & RESIZE ---
    img = cv2.imread(chemin_image)
    if img is None:
        print(f"Erreur lecture: {chemin_image}")
        return

    largeur_cible = 800
    h, w = img.shape[:2]
    ratio = largeur_cible / float(w)
    new_h = int(h * ratio)
    
    img_resized = cv2.resize(img, (largeur_cible, new_h), interpolation=cv2.INTER_AREA)
    output = img_resized.copy() # Image pour dessiner le résultat

    # --- 2. PRÉTRAITEMENT ---
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (9, 9), 2)

    # --- 3. DÉTECTION HOUGH ---
    cercles = cv2.HoughCircles(
        img_blur, cv2.HOUGH_GRADIENT, dp=1,
        minDist=55,
        param1=60, 
        param2=41, 
        minRadius=25, 
        maxRadius=80,
    )

    if cercles is not None:
        cercles = np.round(cercles[0, :]).astype("int")
        print(f"{len(cercles)} pièces détectées.")

        for (x, y, r) in cercles:
            # --- 4. EXTRACTION ROI (Region Of Interest) ---
            # On vérifie qu'on ne sort pas de l'image
            y1, y2 = max(0, y-r), min(new_h, y+r)
            x1, x2 = max(0, x-r), min(largeur_cible, x+r)
            
            roi = img_resized[y1:y2, x1:x2]
            
            # Sécurité : Si la ROI est vide ou bizarre, on saute
            if roi.shape[0] == 0 or roi.shape[1] == 0: continue

            # --- 5. CLASSIFICATION COULEUR ---
            label_metal, couleur = classifier_expert(roi)

            # --- 6. AFFICHAGE ---
            # Cercle autour de la pièce
            cv2.circle(output, (x, y), r, (0, 255, 0), 2)
            
            # Texte (Type de métal + Rayon pour debug)
            texte = f"{label_metal} (r={r})"
            cv2.putText(output, texte, (x-40, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, couleur, 2)

        # Afficher le résultat final
        cv2.imshow("Resultat Final", output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return len(cercles)
    else:
        print("Aucune pièce trouvée. Ajustez param2 ou minDist.")
        return 0
    

# Lancer

detecter_et_identifier("img_pieces/99.png")
