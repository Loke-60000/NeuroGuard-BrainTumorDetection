import numpy as np
import matplotlib.pyplot as plt
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
            
            images.append(img)
            labels.append(label)

    return images, labels

# Example usage:
train = "data/train"
test = "data/test"
val = "data/val"
images, labels = load_data(train)

load_data(train)
plt.imshow(images[12])
plt.title(f"label : {labels[12]}")


print("Nombre total d'images chargées :", len(images))
# print("Nombre total de labels chargés :", len(labels))