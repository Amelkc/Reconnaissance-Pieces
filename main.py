from evaluation.test import *
from evaluation.evaluation import *
from config import IMG_FOLDER, IMAGES_TEST

if __name__=="__main__":
    #tests utiles pour rapport et readme
        #test une seule
    print("TEST SUR UNE SEULE IMAGE : \n ")
    image_test = f"{IMG_FOLDER}/42.jpg"
    tester_unique(imagePath=image_test)
    
        # TEST SUR PLUSIEURS
    print("TEST SUR PLUSIEURS IMAGES : \n ")
    total, nb_images = tester_sur_liste(IMAGES_TEST, afficher_chaque=False)
    moyenne = total / nb_images
    print(f"\nMoyenne: {moyenne:.2f} pièces par image")
    
    #evaluation finale
    print("\n[ÉVAL GT] data.csv")
    evaluer("data/data.csv", afficher=False) #true pour visualiser