import cv2 as cv
import numpy as np
from utils import *

def detecte_pieces(img):
    #pré-traitement : niveau de gris + flou léger (voir si on modifie ça plus tard)
    nvGris = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #on utilisera ca gris+floue
    gaussGray = cv.GaussianBlur(nvGris, (7, 7), 0)
    
    #seuillage adpatatif => les 2 derniers parametres à ajuster  (= taille du voisinage(199) et constate à soustraire(5))
    seuillee = cv.adaptiveThreshold(gaussGray, 255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,41, 8)
    
    # morpho pour nettoyer, remplir les trous, fusionner les bords
    #on prend une ellipse vu quon cherche des pieces mais pk pas tester autre chose
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    opened = cv.morphologyEx(seuillee, cv.MORPH_OPEN, kernel, iterations=1)

    show_image(opened, "nettoyée")
    #attention au chevauchement a réadapter surmement
    num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(opened, connectivity=8)
    
    nb_pieces = 0
    aire_min = 800 # pour retirer bruits parasites mais dcp adapter en fonction taille image => resize toutes les images ???
    for i in range(1, num_labels):
        area = stats[i, cv.CC_STAT_AREA]
        if area >= aire_min:
            nb_pieces += 1
    dessin_cercle(img,num_labels, stats, centroids, aire_min)
    return nb_pieces


def detectePieceContour(img):
    #pré-traitement : niveau de gris + flou léger (voir si on modifie ça plus tard)
    nvGris = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #on utilisera ca gris+floue
    gaussGray = cv.GaussianBlur(nvGris, (5, 5), 0)
    edges = cv.Canny(gaussGray, 30, 150)
    dilated = cv.dilate(edges, (1,1), iterations = 2)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7,7))
    edges = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel)

    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    nb_pieces = 0
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area > 800:
            nb_pieces += 1
            
    return nb_pieces

#img = cv.imread("/Users/amelkaci/MASTER/S2/AnalyseImage/Reconnaissance-Pieces/img_pieces/94.jpg")
img = load_safe_cv2("/Users/amelkaci/MASTER/S2/AnalyseImage/Reconnaissance-Pieces/img_pieces/12.png")
res = detectePieceContour(img)
print(res)