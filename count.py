import cv2 as cv
import numpy as np

def detecte_pieces(img):
    #pré-traitement : niveau de gris + flou léger (voir si on modifie ça plus tard)
    nvGris = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #on utilisera ca gris+floue
    gaussGray = cv.GaussianBlur(nvGris, (7, 7), 0)
    
    #seuillage adpatatif => les 2 derniers parametres à ajuster  (= taille du voisinage(199) et constate à soustraire(5))
    seuillee = cv.adaptiveThreshold(gaussGray, 255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,199, 5)
    
    # morpho pour nettoyer, remplir les trous, fusionner les bords
    #on prend une ellipse vu quon cherche des pieces mais pk pas tester autre chose
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    opened = cv.morphologyEx(seuillee, cv.MORPH_OPEN, kernel, iterations=1)
    closed = cv.morphologyEx(opened, cv.MORPH_CLOSE, kernel, iterations=2)
    
    #attention au chevauchement a réadapter surmement
    num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(closed, connectivity=8)
    
    nb_pieces = 0
    aire_min = 500 # pour retirer bruits parasites mais dcp adapter en fonction taille image => resize toutes les images ???
    for i in range(1, num_labels):
        area = stats[i, cv.CC_STAT_AREA]
        if area >= aire_min:
            nb_pieces += 1
            
    return nb_pieces