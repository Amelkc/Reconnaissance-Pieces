import cv2
import numpy as np
import pandas as pd
from itertools import product
import os

def detect_number_of_coins(chemin_image, dp, minDist, param1, param2, minRadius, maxRadius, blur_ksize):
    """
    Fonction modifiée pour retourner le nombre de pièces détectées sans afficher les images.
    """
    img = cv2.imread(chemin_image)
    if img is None:
        print(f"Erreur: Impossible de lire {chemin_image}")
        return 0

    # RESIZE (standardisation à 800px de largeur)
    taille = 800
    h, w = img.shape[:2]
    ratio = taille / float(w)
    nouvelle_hauteur = int(h * ratio)
    img_resized = cv2.resize(img, (taille, nouvelle_hauteur), interpolation=cv2.INTER_AREA)

    # NIVEAUX DE GRIS
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    # FLOU GAUSSIEN
    img_blur = cv2.GaussianBlur(img_gray, (blur_ksize, blur_ksize), 2)

    # TRANSFORMÉE DE HOUGH
    cercles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, dp=dp, minDist=minDist,
                               param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)

    if cercles is not None:
        cercles = np.round(cercles[0, :]).astype("int")
        return len(cercles)
    else:
        return 0

# Chargement des vérités terrain
df = pd.read_csv('data.csv', header=1)

# Dossier des images (adapte si nécessaire)
dossier_images = "img_pieces/"

# Définition des plages de paramètres pour la recherche en grille
#dp_values = [1.0, 1.2, 1.5]
#minDist_values = [40, 50, 60]
#param1_values = [40, 50, 60]
#param2_values = [25, 30, 35]
#minRadius_values = [20, 30, 40]
#maxRadius_values = [70, 80, 90]

## 64 combinaisons
#dp_values        = [1.0, 1.2]          # 2
#minDist_values   = [45, 55]            # 2
#param1_values    = [40, 60]            # 2
#param2_values    = [25, 35]            # 2
#minRadius_values = [25, 40]            # 2
#maxRadius_values = [70, 90]            # 2

# Version ultra-rapide pour voir si on est sur la bonne voie
#dp_values        = [1.2]
#minDist_values   = [50]
#param1_values    = [50]
#param2_values    = [25, 30, 35]
#minRadius_values = [25, 40]
#maxRadius_values = [75, 90]


## 128 combinaisons : 

#dp_values = [1.0, 1.3]          # 2
#minDist_values = [45, 60]           # 2
#param1_values = [40, 65]          # 2
#param2_values = [25, 38]             # 2
#minRadius_values = [25, 42]             # 2
#maxRadius_values = [68, 92]           # 2
#blur_ksize = [7, 11]               # 2


## Affinage avec 486 tests pour le meilleur trouvé avec 128 : 
dp_values        = [1.0]
minDist_values   = [55, 60, 65]
param1_values    = [60, 65, 70]
param2_values    = [35, 38, 41]
minRadius_values = [23, 25, 28]
maxRadius_values = [88, 92, 96]
blur_ksize       = [9, 11]               # on garde les deux plus performants



# Génération de toutes les combinaisons
all_combos = list(product(dp_values, minDist_values, param1_values, param2_values, minRadius_values, maxRadius_values, blur_ksize))

# Initialisation des meilleurs résultats
best_accuracy = 0
best_params = None
total_images = len(df)

print(f"Recherche en grille sur {len(all_combos)} combinaisons de paramètres...")
print(f"Nombre total d'images : {total_images}")

for idx, combo in enumerate(all_combos):
    dp, minDist, param1, param2, minRadius, maxRadius, blur_ksize = combo
    correct_detections = 0

    for index, row in df.iterrows():
        nom_image = row['Nom image']
        chemin_image = os.path.join(dossier_images, nom_image)
        
        if not os.path.exists(chemin_image):
            print(f"Image manquante : {chemin_image}")
            continue
        
        gt_pieces = int(row['Nombre de pièces'])
        detected = detect_number_of_coins(chemin_image, dp, minDist, param1, param2, minRadius, maxRadius, blur_ksize)
        
        if detected == gt_pieces:
            correct_detections += 1

    accuracy = correct_detections / total_images
    print(f"Combinaison {idx+1}/{len(all_combos)} : {combo} | Accuracy : {accuracy:.2f}")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_params = combo

print("\nMeilleurs paramètres trouvés :")
print(f"dp={best_params[0]}, minDist={best_params[1]}, param1={best_params[2]}, param2={best_params[3]}, minRadius={best_params[4]}, maxRadius={best_params[5]}, blur_ksize={best_params[6]}")
print(f"Accuracy : {best_accuracy:.2f} ({int(best_accuracy * total_images)} images correctes sur {total_images})")
