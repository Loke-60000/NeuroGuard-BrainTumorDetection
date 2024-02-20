import os
import shutil
import numpy as np

# Chemins des dossiers source
dossiers_source = ["brain_tumor_dataset/yes", "brain_tumor_dataset/no"]

# Paramètres configurables
dossier_base = "data"  # Dossier de base pour les dossiers de destination
repartition = {"train": 0.8, "test": 0.1, "val": 0.1}

# Créer tous les dossiers nécessaires
for type_dossier in ["train", "test", "val"]:
    for classe in ["yes", "no"]:
        os.makedirs(os.path.join(dossier_base, type_dossier, classe), exist_ok=True)

# Parcourir les dossiers source
for dossier_source in dossiers_source:
    fichiers = [f for f in os.listdir(dossier_source) if f.endswith('.jpg') or f.endswith('.png')]  # Récupérer uniquement les images
    np.random.shuffle(fichiers)  # Mélanger les fichiers
    
    # Renommer les fichiers dans le dossier 'no'
    if 'no' in dossier_source:
        for i, fichier in enumerate(fichiers):
            os.rename(os.path.join(dossier_source, fichier), os.path.join(dossier_source, 'N' + str(i+1) + '.jpg'))
        fichiers = os.listdir(dossier_source)  # Mettre à jour la liste des fichiers
    
    # Calculer les indices de répartition
    idx_train = int(repartition["train"] * len(fichiers))
    idx_test = idx_train + int(repartition["test"] * len(fichiers))
    
    # Répartir les fichiers
    for i, fichier in enumerate(fichiers):
        chemin_source = os.path.join(dossier_source, fichier)
        
        # Déterminer le dossier de destination
        if i < idx_train:
            dossier_dest = os.path.join(dossier_base, "train", os.path.basename(dossier_source))
        elif i < idx_test:
            dossier_dest = os.path.join(dossier_base, "test", os.path.basename(dossier_source))
        else:
            dossier_dest = os.path.join(dossier_base, "val", os.path.basename(dossier_source))
        
        # Copier le fichier
        shutil.copy2(chemin_source, os.path.join(dossier_dest, fichier))