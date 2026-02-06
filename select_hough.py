import cv2
import numpy as np

def detecter_pieces_optimise(chemin_image):
    """
    Pipeline intégrant :
    1. Resize (Standardisation)
    2. Conversion Gris
    3. Flou Gaussien (Nettoyage)
    4. Transformée de Hough (Détection)
    """
    
    img = cv2.imread(chemin_image)
    if img is None:
        print(f"Erreur: Impossible de lire {chemin_image}")
        return


    # RESIZE
    taille = 800
    h, w = img.shape[:2]
    ratio = taille / float(w)
    nouvelle_hauteur = int(h * ratio)
    
    img_resized = cv2.resize(img, (taille, nouvelle_hauteur), interpolation=cv2.INTER_AREA)
    
    res = img_resized.copy()

    # NIVEAUX DE GRIS
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    # FLOU GAUSSIEN 
    img_blur = cv2.GaussianBlur(img_gray, (9, 9), 2)

    # HOUGH 
    # dp = résulution (1 = mm que l'image)
    # minDist = distance minimale entre les centres de cercles détectés (évite les doublons)
    # param1 = seuil pour Canny (détection de bords)
    # param2 = seuil pour la validation des cercles (plus bas = plus de cercles détectés, mais plus de faux positifs)
    # minRadius & maxRadius = taille attendue des pièces (à ajuster selon les images)

    cercles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, dp=1, minDist=50, param1=50, param2=30, minRadius=30, maxRadius=80)

    # VISUALISATION & RESULTATS
    if cercles is not None:
        cercles = np.round(cercles[0, :]).astype("int")
        print(f"Succès : {len(cercles)} pièces détectées.")
        
        for (x, y, r) in cercles:
            # Cercle extérieur (Vert)
            cv2.circle(res, (x, y), r, (0, 255, 0), 3)
            # Centre (Rouge)
            cv2.circle(res, (x, y), 2, (0, 0, 255), 3)
            
        # Affichage (Appuyez sur une touche pour fermer)
        cv2.imshow("Detection Hough", res)
        
        # Astuce : Afficher aussi ce que l'ordi "voit" pour comprendre les erreurs
        # cv2.imshow("Vue Ordinateur (Blur)", img_blur)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Aucune pièce détectée. Essayez de baisser 'param2'.")

# Utilisation

detecter_pieces_optimise("img_pieces/74.jpg")
