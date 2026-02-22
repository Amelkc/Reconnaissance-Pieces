from coinDetector import *
# TEST UTILES POUR RÉDACTION RAPPORT

IMAGES_TEST = ["img_pieces/12.png","img_pieces/94.jpg","img_pieces/99.png"]

def tester_sur_liste(liste_images=IMAGES_TEST, afficher_chaque=False):
    """
    pour tester la détection sur quelques images 
    """
    total_detectees = 0
    images_testees = 0
    for chemin in liste_images:
        try:
            resultat = detecter_et_identifier(chemin, afficher=afficher_chaque)
            if resultat is not None:
                total_detectees += resultat['nombre']
                images_testees += 1
                # afficher les types détectés
                if resultat['nombre'] > 0:
                    types_detectes = [r['type'] for r in resultat['resultats']]
                    print(f"  Types: {', '.join(types_detectes)}")
        except Exception as e:
            print(f"Erreur lors du traitement de {chemin}: {e}")
            
    print(f"RÉSUMÉ: {total_detectees} pièces détectées sur {images_testees} images")
    return total_detectees, images_testees

def tester_unique(imagePath):
    "test pour avoir le détail des pièces détectées dans le l'image (type, pos,rayon, val)"
    resultat = detecter_et_identifier(imagePath, afficher=True)

    if resultat and resultat['nombre'] > 0:
        print(f"{resultat['nombre']} pièce(s) détectée(s):")
        for i, r in enumerate(resultat['resultats'], 1):
            print(f"  Pièce {i}: {r['type']} \n\t Position: {r['position']}, Rayon: {r['rayon']}px, Valeur: {r['valeur']}€") 

