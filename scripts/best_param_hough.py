import cv2
import numpy as np
import pandas as pd
from itertools import product
import os


"""
Ce script a pour but de rechercher les meilleurs paramètres à utiliser pour le prepocess des images et la transformée de Hough
pour la détection des pièces. 
Nous testons différentes grilles au fur et à mesure et affinons à chaque fois la recherche autour des paramètres utilisés
pour la meilleurs accuracy trouvée à chaque test de grille.
Selon les grilles, le programme peut effectuer une centaine de tests.
Pour 486 tests, le programme a tourné pendant environ 1h30.
"""


def detect_number_of_coins(chemin_image, dp, minDist, param1, param2, minRadius, maxRadius, blur_ksize):
    """
    Cette fonction renvoie le nombre de pièces détectées dans une image avec Hough, sans afficher les images.

    Cette fonction lit l'image, la redimensionne, la convertit en niveaux de gris, applique un flou gaussien,
    puis utilise la méthode Hough pour détecter les cercles (représentant les pièces).

    Args:
        chemin_image (str): chemin complet vers l'image à analyser.
        dp (float): coorespond à la résolution inverse de l'accumulateur (typiquement autour de 1.0-1.5).
        minDist (int): distance minimale entre les centres des cercles détectés.
        param1 (int): seuil pour le détecteur de bord Canny interne.
        param2 (int): seuil pour l'accumulateur de cercles (plus bas = plus de faux positifs).
        minRadius (int): rayon minimum des cercles à détecter (en pixels).
        maxRadius (int): rayon maximum des cercles à détecter (en pixels).
        blur_ksize (int): taille du noyau pour le flou gaussien (doit être impair).

    Returns:
        int: Nb de pièces (donc des cercles) détectées, retourne 0 en cas d'erreur ou si aucun cercle n'est trouvé.
    """
    img = cv2.imread(chemin_image)
    if img is None:
        print(f"Erreur: Impossible de lire {chemin_image}")
        return 0

    # Resize l'image (standardisation à 800px de largeur)
    taille = 800
    h, w = img.shape[:2]
    ratio = taille / float(w)
    nouvelle_hauteur = int(h * ratio)
    img_resized = cv2.resize(img, (taille, nouvelle_hauteur), interpolation=cv2.INTER_AREA)

    # Met l'image en gris
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    # Applique un flou gaussien sur l"image
    img_blur = cv2.GaussianBlur(img_gray, (blur_ksize, blur_ksize), 2)

    # Applique la transformée de Hough
    cercles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, dp=dp, minDist=minDist,
                               param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)

    if cercles is not None:
        cercles = np.round(cercles[0, :]).astype("int")
        return len(cercles)
    else:
        return 0

# Chargement des vérités terrain
# Servira à calculer l'accuracy de la détection des pièces
df = pd.read_csv('data.csv', header=1)

# Dossier des images
dossier_images = "img_pieces/"

# Définition des plages de paramètres pour la recherche en grille
# 1ère grille testée
#dp_values = [1.0, 1.2, 1.5]
#minDist_values = [40, 50, 60]
#param1_values = [40, 50, 60]
#param2_values = [25, 30, 35]
#minRadius_values = [20, 30, 40]
#maxRadius_values = [70, 80, 90]

## 2ème grille testée
#dp_values        = [1.0, 1.2]
#minDist_values   = [45, 55]
#param1_values    = [40, 60]
#param2_values    = [25, 35]
#minRadius_values = [25, 40]
#maxRadius_values = [70, 90]

# 3ème grille testée
#dp_values        = [1.2]
#minDist_values   = [50]
#param1_values    = [50]
#param2_values    = [25, 30, 35]
#minRadius_values = [25, 40]
#maxRadius_values = [75, 90]


## 128 combinaisons, 4ème grille

#dp_values = [1.0, 1.3]
#minDist_values = [45, 60]
#param1_values = [40, 65]
#param2_values = [25, 38]
#minRadius_values = [25, 42]
#maxRadius_values = [68, 92]
#blur_ksize = [7, 11]


## Affinage avec 486 tests pour le meilleur trouvé avec 128 (4ème grille) : 
#dp_values        = [1.0]
#minDist_values   = [55, 60, 65]
#param1_values    = [60, 65, 70]
#param2_values    = [35, 38, 41]
#minRadius_values = [23, 25, 28]
#maxRadius_values = [88, 92, 96]
#blur_ksize       = [9, 11]


## Grille de 729 tests pour affiner autour des paramètres de 0.68 d'accuracy.
# Résultat : 0.71 d'accuracy
#dp_values        = [1.0]
#minDist_values   = [60, 65, 70]           # autour de 65
#param1_values    = [62, 65, 68]           # léger zoom
#param2_values    = [39, 41, 43]           # autour de 41
#minRadius_values = [26, 28, 30]           # autour de 28
#maxRadius_values = [92, 96, 100]          # un peu plus large
#blur_ksize       = [7, 9, 11]             # on remet 7 en jeu



# Dernière grille testée, pour affiner autour de 0.71
# Résultat : 0.72 d'accuracy. Dernière grille testée.
dp_values = [1.0]
minDist_values = [68, 70, 72]
param1_values = [60, 62, 64]
param2_values = [41, 43, 45]
minRadius_values = [24, 26, 28]
maxRadius_values = [98, 100, 102]
blur_ksize = [9, 11]


# Génération de toutes les combinaisons de paramètres à tester pour la détection des pièces
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
        
        gt_pieces = int(row['Nombre de pièces']) #nb réel de pièces (groud truth)
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



