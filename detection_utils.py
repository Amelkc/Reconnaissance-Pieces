import numpy as np
import cv2
from preproc import *


def detecter_pieces_hough(img):
    """
    détection de pièces avec Hough
    """
    preprocessed = pretraitement(img)
    # on va utilsier les meilleurs paramètres qu'on avait trouvé avec best_param_hough
    circles = cv2.HoughCircles(
        preprocessed,
        cv2.HOUGH_GRADIENT,
        dp=1.0,
        minDist=70,
        param1=64,  
        param2=43, 
        minRadius=26,
        maxRadius=102
    )

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        return circles
    return None



def detecter_pieces_watershed(img):
    """
    détection par segmentation watershed
    """
    preprocessed = pretraitement(img)
    
    #on binarise avec Otsu
    ret, thresh = cv2.threshold(preprocessed, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # pour retirer le bruit
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    dilated = cv2.dilate(opening, kernel, iterations=3)

    #pour mieux identifier les bords des pieces, bien distinguer du fond
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, piece_sur = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)

    #on va identifier les regions inconnues (ca peut etre un bord de piece comme appartenir au fond on sait pas trop)
    piece_sur = np.uint8(piece_sur)
    inconnu = cv2.subtract(dilated, piece_sur)

    #composantes connexes
    ret, labels = cv2.connectedComponents(piece_sur)
    labels = labels + 1
    labels[inconnu == 255] = 0

    # watershed (attetion avec img couleur)
    labels = cv2.watershed(img, labels)

    #recuperer les cercles
    circles = []
    labels_unique = np.unique(labels)
    
    for label in labels_unique:
        if label <= 1: continue # 0=frontière, 1=fond
        mask = np.zeros(preprocessed.shape, dtype=np.uint8)
        mask[labels == label] = 255
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(cnts) > 0:
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            area = cv2.contourArea(c)
            if 20 < radius < 110 and area > 600: 
                perimeter = cv2.arcLength(c, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity > 0.75: 
                        circles.append((int(x), int(y), int(radius)))

    return np.array(circles) if len(circles) > 0 else None



def fusionner_cercles(circles1, circles2, dist_k=0.35, rtol=0.25, prefer_first=True):
    """
    fusionne deux listes de cercles (avec élimination des doublons).
    prefer_first pour etre sur qu'on fait bien un choix si égalité (true alors on garde celui de C1)
    """
    if circles1 is None and circles2 is None:
        return None
    if circles1 is None:
        return np.array(list(circles2))
    if circles2 is None:
        return np.array(list(circles1))

    c1_list = [tuple(map(int, c)) for c in list(circles1)]
    c2_list = [tuple(map(int, c)) for c in list(circles2)]

    merged = c1_list.copy()
    
    for (x2, y2, r2) in c2_list:
        # attention doublon si centres proches et rayons proches
        def is_doublon(cercleI):
            i, (x1, y1, r1) = cercleI
            dx, dy = x1 - x2, y1 - y2
            dist = (dx*dx + dy*dy) ** 0.5
            return dist < dist_k * min(r1, r2) and abs(r1 - r2) < rtol * max(r1, r2)
        
        duplicate = next(filter(is_doublon, enumerate(merged)), None)
        if duplicate is None:
            merged.append((x2, y2, r2))
        else:
            #si on ne préfère pas C1, on peut garder le plus "grand"
            if not prefer_first:
                duplicate_idx, (x1, y1, r1) = duplicate
                if r2 > r1:
                    merged[duplicate_idx] = (x2, y2, r2)

    return np.array(merged, dtype=int)

