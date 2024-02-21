import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

def load_data(path, target_size=(224, 224)):
    """
    Charge les images à partir du chemin spécifié et les redimensionne à la taille cible.
    
    :param path: Chemin vers le répertoire contenant les images.
    :param target_size: Taille cible des images (par défaut : (224, 224) pour VGG16).
    :return: Tableaux NumPy des images et des étiquettes correspondantes.
    """
    images = []
    labels = []

    for folder_name in os.listdir(path):
        folder_path = os.path.join(path, folder_name)
        label = 1 if folder_name == "yes" else 0  # assigner 1 aux images dans le dossier "yes", sinon 0
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, target_size)  # redimensionner l'image à la taille cible
            images.append(img)
            labels.append(label)

    return np.array(images), np.array(labels)

