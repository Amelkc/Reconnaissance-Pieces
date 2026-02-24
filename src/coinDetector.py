import cv2
from src.utils import *
from src.detection_utils import *
from src.classify import *

def detecter_et_identifier(chemin_image, afficher=True):
    """
    combine Hough et segmentation watershed pour detecter les pieces dans une image
    """
    img = load_safe_cv2(chemin_image)
    if img is None:
        print(f"Erreur lecture: {chemin_image}")
        return None

    # attention pas la même échelle pour toutes les images => redimensionner
    largeur_cible = 800
    h, w = img.shape[:2]
    ratio = largeur_cible / float(w)
    new_h = int(h * ratio)
    img_resized = cv2.resize(img, (largeur_cible, new_h), interpolation=cv2.INTER_AREA)
    output = img_resized.copy()
    #partie détection
    #hough
    circles_hough = detecter_pieces_hough(img_resized)
    #segmentation wateshed
    circles_seg = detecter_pieces_watershed(img_resized)
    # fusion hough et watershed (et on élimine les doublons)
    circles = fusionner_cercles(circles_hough, circles_seg)

    if circles is not None and len(circles) > 0:
        print(f"{len(circles)} pièces détectées dans {chemin_image}")
        resultats = []

        for (x, y, r) in circles:
            #extraire les ROI
            y1, y2 = max(0, y-r), min(new_h, y+r)
            x1, x2 = max(0, x-r), min(largeur_cible, x+r)
            roi = img_resized[y1:y2, x1:x2]

            if roi.shape[0] == 0 or roi.shape[1] == 0:
                continue

            #partie classification
            label_metal, couleur = classifier_piece(roi)

            # Si la classification est "Inconnu", on la rejette.
            if label_metal == "Inconnu":
                continue # On ignore cette détection car probablement du bruit (bois, ombre)

            resultats.append({
                'position': (x, y),
                'rayon': r,
                'type': label_metal
            })

            label_metal, couleur = classifier_piece(roi)
            valeur = get_valeur_piece(label_metal, r)
            resultats[-1]['valeur'] = valeur 

            #préparer les cercles sur l'image pour visualiser
            cv2.circle(output, (x, y), r, (0, 255, 0), 2)
            texte = f"{label_metal} (r={r})"
            cv2.putText(output, texte, (x-40, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, couleur, 2)

        if afficher:
            cv2.imshow("Resultat Final", output)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        
        somme_totale = sum(r['valeur'] for r in resultats)

        return {
            'nombre': len(circles),
            'resultats': resultats,
            'image': output,
            'somme': somme_totale
        }
    else:
        print(f"Aucune pièce trouvée dans {chemin_image}")
        #pas renvoyer None pour les calcul stats plus tard
        return {
            'nombre': 0,
            'resultats': [],
            'image': output,
            'somme' : 0
        }
