# chemins d'accès aux modèles et aux ressources
IMG_FOLDER = "data/img_pieces/"
IMAGES_TEST = ["data/img_pieces/12.png","data/img_pieces/94.jpg","data/img_pieces/99.png"]


# couleurs de référence (BGR) pour l'affichage
COLOR_1EURO = (255, 0, 0) #bleu
COLOR_2EUROS = (0, 255, 0) #vert
COLOR_CUIVRE = (0, 0, 255) #rouge pour les pièces en cuivre (centimes)
COLOR_OR = (0, 255, 255) #jaune pour les pièces en or (centimes)


# paramètres Watershed
WATERSHED_MIN_AREA = 600
WATERSHED_CIRCULARITY = 0.75

# seuils Canny / Hough
CANNY_THRESHOLD_1 = 64
CANNY_THRESHOLD_2 = 43
MIN_RADIUS = 26
MAX_RADIUS = 102