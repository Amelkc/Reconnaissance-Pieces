# Détection et identification de pièces de monnaies en euros - Projet M1 Vision et Machines Intelligentes


## Description du projet

Ce projet implémente un système automatique de détection, identification et comptage de pièces d'euros sur des images. Le système combine des techniques de traitement d'image classique (transformée de Hough, segmentation Watershed) avec une classification basée sur l'analyse de couleur pour identifier le type et la valeur de chaque pièce détectée.


### Objectif principaux : 

- Détecter automatiquement les pièces circulaires sur une image

- Identifier le type de pièce (cuivre, or, 1€, 2€)

- Estimer la valeur monétaire en euros à partir du rayon et de la couleur

- Calculer la somme totale présente sur l'image

- Évaluer les performances avec des métriques quantitatives



## Structure du projet

```bash
projet_monnaie/
│
├── main.py                   # Point d'entrée - lance tests et évaluation
├── config.py                 # Configuration (couleurs, paramètres Hough/Watershed)
│
├── src/                      # Code source principal
│   ├── __init__.py           # Fichier d'initialisation du module
│   ├── coinDetector.py       # Pipeline complet de détection et identification
│   ├── classify.py           # Classification des pièces par analyse HSV
│   ├── detection_utils.py    # Détection Hough et Watershed + fusion
│   ├── preproc.py            # Prétraitement des images (HSV, CLAHE, flou)
│   └── utils.py              # Utilitaires généraux (chargement, visualisation)
│
├── scripts/                  # Scripts utilitaires et optimisation
│   ├── __init__.py
│   └── best_param_hough.py   # Recherche des meilleurs paramètres Hough (grille)
│
├── evaluation/               # Tests et métriques de performance
│   ├── __init__.py
│   ├── test.py               # Fonctions de test sur images individuelles/listes
│   └── evaluation.py         # Calcul des métriques (MAE, RMSE, MAPE)
│
├── data/                     # Données du projet
│   ├── img_pieces/           # Dossier pour stocker les images
│   └── data.csv              # Fichier CSV avec ground truth (nom, nb pièces, valeur)
│
├── requirements.txt          # Dépendances Python
└── README.md                 # Ce fichier
```



## Installation

### Prérequis

- Python 3.8+
- pip

### Installation des dépendances

```bash
pip install -r requirements.txt
```

### Dépendances principales : 

- opencv-python : traitement d'image et détection

- numpy : calculs numériques

- pandas : manipulation de données CSV

- matplotlib : visualisation

- Pillow : chargement d'images



## Utilisation

Lancement du programme principal
Depuis la racine du projet, exécutez :

```bash
python main.py
```

### Ce que fait main.py : 

1. Test sur une image unique : détecte et affiche les pièces sur img_pieces/98.png

2. Test sur plusieurs images : traite une liste d'images de test et calcule la moyenne de pièces détectées

3. Évaluation complète : compare les résultats avec le fichier data.csv (ground truth) et affiche les métriques



## Description des modules

### src/coinDetector.py - Pipeline principal

- Fonction principale : **detecter_et_identifier(chemin_image, afficher=True)**

    Combine la détection (Hough + Watershed) et la classification pour identifier toutes les pièces d'une image.

    Étapes :

    1. Chargement et redimensionnement de l'image (largeur = 800px)

    2. Détection des pièces avec **Hough** (transformée de cercles)

    3. Détection des pièces avec **Watershed** (segmentation)

    4. Fusion des deux méthodes pour éliminer les doublons

    5. Pour chaque pièce détectée :

        - Extraction de la région d'intérêt (ROI)

        - Classification du type (cuivre, or, 1€, 2€)

        - Estimation de la valeur monétaire

    6. Affichage des résultats avec cercles et labels

    **Sortie :**

    ```bash
    {
        'nombre': 8,  # Nombre de pièces détectées
        'resultats': [  # Liste des pièces
            {'position': (x, y), 'rayon': r, 'type': 'Cuivre (1,2,5c)', 'valeur': 0.05},
            ...
        ],
        'image': array(...),  # Image annotée
        'somme': 3.87  # Somme totale en euros
    }
    ```


### src/detection_utils.py - Détection de cercles

- Fonction : **detecter_pieces_hough(img)**
    Utilise la transformée de Hough pour détecter les cercles (pièces).

    Paramètres optimisés (trouvés par grid search) :

    - dp=1.0 : résolution de l'accumulateur

    - minDist=70 : distance minimale entre centres

    - param1=64 : seuil Canny haut

    - param2=50 : seuil d'accumulation

    - minRadius=26, maxRadius=102 : plage de rayons

- Fonction : **detecter_pieces_watershed(img)**
    Utilise l'algorithme de segmentation Watershed pour détecter les pièces.

    Étapes :

    1. Binarisation avec seuillage d'Otsu

    2. Opérations morphologiques (ouverture, dilatation) pour retirer le bruit

    3. Transformée de distance pour identifier les centres des pièces

    4. Segmentation Watershed pour séparer les pièces qui se touchent

    5. Extraction des contours et calcul de la circularité (filtre > 0.75)

- Fonction : **fusionner_cercles(circles_hough, circles_watershed, ...)**
    Fusionne les résultats de Hough et Watershed en éliminant les doublons (cercles détectés par les deux méthodes).

    Critères de doublon :

    - Distance entre centres < 0.6 × min(rayon1, rayon2)

    - Différence de rayons < 0.3 × max(rayon1, rayon2)

    Stratégie : Par défaut, privilégie le cercle de Hough en cas d'égalité (prefer_first=True).


### src/classify.py - Classification des pièces

- Fonction : **classifier_piece(roi_img)**
    Classifie une pièce en analysant sa couleur HSV dans deux zones :

    - Cœur : zone centrale (40% du rayon)

    - Couronne : zone périphérique (entre 60% et 90% du rayon)

    Prétraitement de la ROI :

    1. Conversion en espace LAB

    2. Égalisation d'histogramme adaptatif (CLAHE) sur le canal L

    3. Retour en BGR puis conversion en HSV

    Règles de classification :

    - -Cuivre (1c, 2c, 5c) : teinte rouge (H < 15 ou H > 165) + saturation > 30

    - 1 Euro : différence de saturation > 15 entre cœur et couronne (cœur moins saturé)

    - 2 Euros : teinte jaune-orange (15 < H < 40)

    - Or (10c, 20c, 50c) : saturation uniforme, pas de rouge dominant

    **Sortie :**

    ```bash
    ("1 EURO", (255, 0, 0))  # (label, couleur BGR pour affichage)
    ```


- Fonction : **get_valeur_piece(label, rayon_px)**
    Associe une valeur monétaire à partir du type et du rayon.

    Seuils de rayon (en pixels, pour image redimensionnée à 800px) :

    - Cuivre : rayon < 50 → 0.01€ | < 70 → 0.02€ | sinon → 0.05€

    - Or : rayon < 60 → 0.10€ | < 80 → 0.20€ | sinon → 0.50€

    - 1 Euro : 1.00€

    - 2 Euros : 2.00€


### src/preproc.py - Prétraitement

- Fonction : **pretraitement(img)**
    Prépare l'image pour la détection de cercles.

    Pipeline :

    1. Conversion en espace HSV (plus robuste aux variations d'éclairage que RGB)

    2. Extraction du canal V (Value = luminosité). Le canal V est invariant aux changements de teinte/saturation, idéal pour détecter les formes circulaires indépendamment de la couleur.

    3. CLAHE (Contrast Limited Adaptive Histogram Equalization) pour normalisation locale

        - clipLimit=1.5, tileGridSize=(8,8)

    4. Flou gaussien (kernel 13×13) pour réduire le bruit

    **Sortie :** Image en niveaux de gris normalisée, prête pour Hough ou Watershed.


### evaluation/test.py - Tests

- Fonction : **tester_sur_liste(liste_images, afficher_chaque=False)**
    Teste la détection sur plusieurs images et retourne le nombre total de pièces détectées.

    Exemple :

    ```bash
    IMAGES_TEST = ["img_pieces/12.png", "img_pieces/94.jpg", "img_pieces/99.png"]
    total, nb_images = tester_sur_liste(IMAGES_TEST, afficher_chaque=False)
    print(f"Moyenne : {total / nb_images:.2f} pièces/image")
    ```

- Fonction : **tester_unique(imagePath)**
    Affiche les détails complets de détection sur une seule image :

    - Position (x, y) de chaque pièce

    - Rayon en pixels

    - Type identifié

    - Valeur monétaire


### evaluation/evaluation.py - Métriques de performance

- Fonction : **evaluer(csv_path, img_folder="img_pieces/", afficher=False)**
    Compare les prédictions avec la vérité terrain (ground truth) contenue dans data.csv.

- Métriques calculées :
    - **MAE** : Erreur Absolue Moyenne (comptage)
    - **RMSE** : Racine de l'Erreur Quadratique Moyenne
    - **MAPE** : Erreur Moyenne en Pourcentage (valeur €)
    - **Taux exact** : % d'images avec 0 erreur de comptage
    - **Écart-type** : Dispersion des erreurs
    - **Taux de détection** : Ratio pièces détectées/réelles

Format du CSV : 

```bash
Nom image,Nombre de pièces,Valeur monétaire €
12.png,8,3.87
94.jpg,6,4.20
...
```

Sortie console : 

```bash
## TOTAL IMAGES : 106
        COMPTAGE
MAE: 2.06 | RMSE: 6.18
IMAGES SANS ERREUR (%): 71.7%
ECART-TYPE : 5.85
MOYENNE TAUX DE DÉTECTION: 1.01
        SOMME (EN EUROS)
MAE €: 2.71 | RMSE €: 4.08 | MAPE %: 78.4%
ECART-TYPE : 3.06
```


### scripts/best_param_hough.py - Optimisation des paramètres

Ce script effectue une recherche en grille (grid search) pour trouver les meilleurs paramètres de la transformée de Hough. Il a servi au processus de création  des fonctions de détection de pièce.

Principe :

1. Définir des plages de valeurs pour chaque paramètre

2. Générer toutes les combinaisons possibles

3. Pour chaque combinaison :

    - Détecter les pièces sur toutes les images du dataset

    - Comparer avec le ground truth

    - Calculer l'accuracy

4. Retourner la combinaison avec la meilleure accuracy

Exemple de grille testée (729 combinaisons) :

```bash
dp_values = [1.0]
minDist_values = [68, 70, 72]
param1_values = [60, 62, 64]
param2_values = [41, 43, 45]
minRadius_values = [24, 26, 28]
maxRadius_values = [98, 100, 102]
blur_ksize = [9, 11]
```

Résultats : Après plusieurs itérations, les paramètres optimaux ont atteint 72% d'accuracy sur le comptage exact.

**Lancement :**

```bash
python -m scripts.best_param_hough
```


### config.py - Configuration globale

Centralise les constantes utilisées dans le projet.

Couleurs d'affichage (BGR) :

```bash
COLOR_1EURO = (255, 0, 0)      # Bleu
COLOR_2EUROS = (0, 255, 0)     # Vert
COLOR_CUIVRE = (0, 0, 255)     # Rouge (centimes cuivre)
COLOR_OR = (0, 255, 255)       # Jaune (centimes or)
```

Paramètres Watershed :

```bash
WATERSHED_MIN_AREA = 600       # Aire minimale d'un contour
WATERSHED_CIRCULARITY = 0.75   # Seuil de circularité
```

Paramètres Hough :

```bash
CANNY_THRESHOLD_1 = 64
CANNY_THRESHOLD_2 = 43
MIN_RADIUS = 26
MAX_RADIUS = 102
```

### src/utils.py - Fonctions utilitaires

- **load_safe_cv2(path_img)**
Charge une image de manière robuste en passant par PIL pour gérer différents formats.

- **show_image(img, title)**
Affiche une image avec matplotlib (utile pour le débogage).

- **read_data()**
Lit et affiche le contenu du fichier CSV de vérité terrain.

- **dessin_cercle(...)**
Dessine des cercles sur une image pour visualiser les composantes connexes (debug).


## Méthodes et Algorithmes

### Détection par transformée de Hough

Principe : Détecte les formes circulaires en cherchant des accumulations de votes dans un espace de paramètres (centre x, y et rayon r).

Avantages :

- Robuste au bruit et aux variations d'éclairage

- Rapide (implémentation OpenCV optimisée)

- Fonctionne bien avec des pièces isolées

Limites :

- Difficulté à détecter des pièces partiellement visibles

- Paramètres sensibles (nécessite optimisation)

### Détection par segmentation Watershed

Principe : Traite l'image comme une carte topographique et inonde progressivement depuis les minima locaux pour séparer les régions.

Étapes clés :

1. Binarisation d'Otsu pour séparer fond et objets

2. Morphologie mathématique pour nettoyer

3. Transformée de distance pour trouver les centres des pièces

4. Watershed pour segmenter chaque pièce


Avantages :

- Meilleure séparation des pièces qui se touchent

- Moins dépendant des paramètres qu'Hough

Limites :

- Plus sensible au bruit et aux textures du fond

- Plus lent que Hough

En réalité, avec les paramètres optimisés, la méthode de Hough est bien plus efficace que la segmentation par Watershed qui apporte tout de même des precisions sur certaines images.


### Classification par Analyse Couleur HSV

Principe : Analyse la distribution de teinte (Hue) et saturation dans deux zones de la pièce pour distinguer les métaux.

HSV permet une meilleure analyse des médias :

- Teinte (H) : représente la couleur pure (rouge pour cuivre, jaune pour or)

- Saturation (S) : intensité de la couleur (élevée = métal brillant)

- Valeur (V) : luminosité (peu utilisée ici)


Zones analysées :

- Cœur central : 40% du rayon

- Couronne externe : entre 60% et 90% du rayon


Distinction 1€ vs 2€ :

- 1€ : cœur argenté (faible saturation) + couronne dorée (haute saturation)

- 2€ : cœur doré (teinte jaune-orange) + couronne argentée


###  Fusion des Détections

Problème : Hough et Watershed détectent parfois la même pièce → doublons.

Solution : Algorithme de fusion qui compare chaque cercle de Watershed avec ceux de Hough :

- Si centres proches ET rayons similaires c'est un doublon, on garde un seul cercle

- Sinon c'est une nouvelle pièce et on l'ajoute au résultat final


Critères de doublon ajustables :

```bash
dist_k=0.35      # Distance relative au rayon
rtol=0.25        # Tolérance relative sur le rayon
prefer_first=True # Priorité à Hough en cas d'égalité
```



## Résultats et performances

### Performances : 

- MAE (comptage) : 2.06 pièces
    En moyenne, 2.06 pièce d'écart par image

- RMSE (comptage) : 6.18 pièces
    Racine de l'erreur quadratique moyenne

- Taux exact : 71.7%	
    71.7% des images ont un comptage parfait

- MAE (valeur)	2.71 €	
    Erreur moyenne de 2.71 € par image

- MAPE	78.4%	
    Erreur relative de 78.4% sur la valeur

- Taux de détection	1.01	
    Détecte en moyenne 142% des pièces présentes


### Facteur influençant les performances

- Condition favorables : 

    - Fond uni et contrasté

    - Éclairage uniforme

    - Pièces bien espacées

    - Caméra perpendiculaire (pas de perspective)

- Conditions défavorables :

    - Pièces qui se chevauchent

    - Ombres marquées ou reflets

    - Fond texturé ou de couleur similaire aux pièces

    - Pièces partiellement hors cadre




## Améliorations possibles

- Améliorer le prétraitement dans preproc.py (changer clipLimit, kernel de flou)

- Ajuster les seuils de classification dans classify.py (teintes, saturations)

- Implémenter un réseau de neurones (CNN) pour la classification

