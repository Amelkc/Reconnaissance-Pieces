#fonctions utiles pour éviter redondance code et faciliter les tests
import matplotlib.pyplot as plt
import cv2 as cv
from PIL import Image
import numpy as np
import pandas as pd

PATH_DATA='data/data.csv'

def show_image(img, title):
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()
    
def dessin_cercle(img,num_labels, stats, centroids, aire_min):
    #pour debugger mais reverif si cette fonction correcte
    img_debug = img.copy()
    for i in range(1, num_labels):
        area = stats[i, cv.CC_STAT_AREA]
        if area < aire_min:
            continue
        x, y, w, h, _ = stats[i]
        cx, cy = centroids[i]
        # rayon approximatif
        r = int(0.5 * (w + h) / 2)
        cv.circle(img_debug, (int(cx), int(cy)), r, (0, 0, 255), 2)
    show_image(img_debug, "debug")


def load_safe_cv2(path_img):
    img_pil = Image.open(path_img)
    img_bgr = cv.cvtColor(np.array(img_pil), cv.COLOR_RGB2BGR)
    return img_bgr

def read_data():
    df = pd.read_csv(PATH_DATA, header=1)
    print("DATA :")
    print(df.head())
    for index, row in df.iterrows():
        nom_image = row['Nom image']
        nb_pieces_reel = row['Nombre de pièces']
        valeur_reelle = row['Valeur monétaire €']
        print(f"Image : {nom_image} | Vérité : {nb_pieces_reel} pièces, {valeur_reelle} €")
