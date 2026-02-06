import pandas as pd
from class_color import *

NB_ECHANTILLONS=106

if __name__=="__main__":
    df = pd.read_csv('data.csv', header=1)
    #pour garder aperçu perf globales
    predict_nb_TRUE = 0  #predictions correctes NOMbRE DE PIÈCES
    #predict_val_TRUE = 0 #predictions correctes VALEUR MONÉTAIRES
    
    #on va garder une trace de nos images erronées pr le readme
    img_nb_FALSE = []  #contient nom_image dont on a pas su predire le nombre de pieces
    #img_val_FALSE = []  #contient nom_image dont on a pas su predire la valeur monétaire
    for index, row in df.iterrows():
        nom_image = row['Nom image']
        path="img_pieces/" + nom_image
        nb_attendu = int(row['Nombre de pièces'])
        val_attendue = float(row['Valeur monétaire €'])
        
        res_nb = detecter_et_identifier(path)
        if res_nb == nb_attendu :
            predict_nb_TRUE += 1
        else : 
            img_nb_FALSE.append(nom_image)
            
          #______________ A DECOMMMENTER +TARD   
        #res_val = 0
        #if res_val == val_attendue :
        #    predict_val_TRUE += 1
        #else : 
        #    img_val_FALSE.append(nom_image)
        
    acc_nb = predict_nb_TRUE / NB_ECHANTILLONS
    print(f"Accuracy Compte Pièces : {acc_nb:.2f} ({predict_nb_TRUE} images correctes sur {NB_ECHANTILLONS})")
    #acc_val = predict_val_TRUE / NB_ECHANTILLONS
    #print(f"Accuracy Calcul Valeur : {acc_val:.2f} ({predict_val_TRUE} images correctes sur {NB_ECHANTILLONS})")