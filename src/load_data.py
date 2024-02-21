import numpy as np
import cv2
import os

def load_data(path):
    path_yes = os.path.join(path, "yes")
    path_no = os.path.join(path, "no")

    images = []
    labels = []

    for folder_path, label in [(path_yes, 1), (path_no, 0)]:
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (224, 224))  # Redimensionner l'image
                images.append(img)
                labels.append(label)

    images = np.array(images, dtype="float32") / 255.0  # Normalisation
    labels = np.array(labels)
    return images, labels

# # Exemple d'utilisation :
# train_path = "data/train"  # Assurez-vous que ce chemin est correct
# images, labels = load_data(train_path)

# # Affichage d'une image et de son label pour vérification
# plt.imshow(cv2.cvtColor(images[12], cv2.COLOR_BGR2RGB))  # Conversion BGR en RGB pour l'affichage correct
# plt.title(f"label : {labels[12]}")
# plt.show()

# print("Nombre total d'images chargées :", len(images))
# print("Nombre total de labels chargés :", len(labels))
