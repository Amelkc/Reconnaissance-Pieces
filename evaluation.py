import csv
import statistics
import math
import os
from coinDetector import *


def evaluer(csv_path, img_folder="img_pieces/", afficher=False):
    """
    evaluation avec métriques standards : 
    MAE
    RMSE
    MAPE
    ecart-type
    """
    erreurs_count = []
    sq_erreurs_count = []
    erreurs_val = []
    sq_erreurs_val = []
    erreurs_sum_pct = []
    taux_detect = []  # pred/val comptage attendue
    
    # les resultats par image
    resultats = []
    
    with open(csv_path, 'r', encoding='utf-8-sig') as f:  
        reader = csv.reader(f) 
        next(reader, None) 
        
        for row_num, row in enumerate(reader, 2):  
            img_name = row[0].strip()
            
            #nos val attendues
            count_attendu = int(float(row[1]))
            somme_attendue = float(row[2]) 
            
            img_path = os.path.join(img_folder, img_name)
            if not os.path.exists(img_path):
                print(f"Image manquante: {img_path}")
                continue
            
            resultat = detecter_et_identifier(img_path, afficher=afficher)
            
            pred_count = resultat['nombre']
            pred_sum = resultat.get('somme', 0.0)
            
            err_count = abs(pred_count - count_attendu)
            err_sum_abs = abs(pred_sum -  somme_attendue )
            err_sum_pct = (err_sum_abs /  somme_attendue * 100) if  somme_attendue > 0 else 0
            
            erreurs_count.append(err_count)
            erreurs_val.append(err_sum_abs)
            erreurs_sum_pct.append(err_sum_pct)

            sq_erreurs_count.append(err_count ** 2)
            sq_erreurs_val.append(err_sum_abs ** 2)
            taux_detect.append(pred_count / count_attendu if count_attendu > 0 else 0)
            
            resultats.append({
                'Image': img_name,
                'GT Count': count_attendu, 'Pred Count': pred_count, 'Err Count': err_count,
                'GT €': f"{ somme_attendue:.2f}", 'Pred €': f"{pred_sum:.2f}", 'Err €': f"{err_sum_abs:.2f}",
                '% Err': f"{err_sum_pct:.1f}%"
            })
    
    if not resultats:
        print("Aucune image valide.")
        return
    
    # Stats
    mae_count = statistics.mean(erreurs_count)
    rmse_count = math.sqrt(statistics.mean(sq_erreurs_count))
    mae_val = statistics.mean(erreurs_val)
    rmse_sum = math.sqrt(statistics.mean(sq_erreurs_val))
    mape_sum = statistics.mean(erreurs_sum_pct)
    
    # %img avec 0 erreur
    taux_exact = (sum(1 for e in erreurs_count if e == 0) / len(erreurs_count)) * 100
    #voir si on detecte trop de pieces ou pas assez 
    avg_taux_detect = statistics.mean(taux_detect)
    
    
    print("\n## Tableau résultats")   
    print(f"\n## TOTAL IMAGES : {len(resultats)}")
    
    print("\tCOMPTAGE")
    print(f"MAE: {mae_count:.2f} | RMSE: {rmse_count:.2f}")
    print(f"IMAGES SANS ERREUR (%): {taux_exact:.1f}%")
    print(f"ECART-TYPE : {statistics.stdev(erreurs_count):.2f}")
    print(f"MOYENNE TAUX DE DÉTECTION: {avg_taux_detect:.2f}")
    
    print("\tSOMME (EN EUROS)")
    print(f"MAE €: {mae_val:.2f} | RMSE €: {rmse_sum:.2f} | MAPE %: {mape_sum:.1f}%")
    print(f"ECART-TYPE : {statistics.stdev(erreurs_val):.2f}")

