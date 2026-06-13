# Prediction_cdm
Script de prédiction des matchs de la coupe du monde 2026

----------- Si vous voulez utiliser les fichiers de données déjà existant et faire tourner uniquement la prédiction entrez cette commande :
py .\main.py --home "Pays 1" --away "Pays 2"
Exemple : py .\main.py --home "Paraguay" --away "États-Unis"

Attention : vérifier bien l'orthographe du pays (vous pouvez la trouver dans stats_equipe.csv)

----------- Si vous voulez entrainer le modele en changement uniquement les paramètres :

py .\preparation_model.py

Optuna se chargera de mettre les meilleurs paramètres

------------ Si vous voulez lancer la création de tous les fichers :

Exécuter toutes les cellules de Creation_data.ipynb
Exécuter py .\preparation_model.py pour la création du pkl
et éxécuter .\main.py --home "Pays 1" --away "Pays 2"
