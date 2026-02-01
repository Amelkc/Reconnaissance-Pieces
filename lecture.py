import pandas as pd

df = pd.read_csv('data.csv', header=1)


print("DATA :")
print(df.head())


for index, row in df.iterrows():
    nom_image = row['Nom image']
    nb_pieces_reel = row['Nombre de pièces']
    valeur_reelle = row['Valeur monétaire €']
    
    print(f"Image : {nom_image} | Vérité : {nb_pieces_reel} pièces, {valeur_reelle} €")
