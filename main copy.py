from keras.applications import VGG16
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
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


def auto_crop_and_center_image(image_path, output_path, canvas_width=256, canvas_height=256):
    """
    Procède automatiquement au recadrage et au centrage d'une image autour de la région d'intérêt.
    La région d'intérêt est le plus grand contour trouvé dans l'image.
    L'image est redimensionnée pour s'adapter à une taille de canevas spécifiée et centrée.
    L'image finale est enregistrée dans le chemin de sortie spécifié.

    :param image_path: Le chemin de l'image d'entrée.
    :param output_path: Le chemin pour enregistrer l'image traitée.
    :param canvas_width: La largeur du canevas en pixels.
    :param canvas_height: La hauteur du canevas en pixels.
    """
    import numpy as np

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not open or find the image {image_path}")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)

    thresh = cv2.bitwise_not(thresh)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print(f"No contours found in image {image_path}. Image is not processed.")
        return

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    cropped_image = image[y:y+h, x:x+w]

    aspect_ratio = w / h
    if aspect_ratio > 1:
        new_h = int(canvas_height / aspect_ratio)
        new_w = canvas_width
    else:
        new_w = int(canvas_width * aspect_ratio)
        new_h = canvas_height

    new_w, new_h = max(1, new_w), max(1, new_h)

    resized_image = cv2.resize(cropped_image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    background = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

    x_offset = (canvas_width - new_w) // 2
    y_offset = (canvas_height - new_h) // 2

    background[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_image

    cv2.imwrite(output_path, background)
    print(f"Processed and saved: {output_path}")


def process_folder_for_cropping(folder_path, output_folder, canvas_width=256, canvas_height=256):
    """
    Traite toutes les images dans un dossier spécifié, les recadrant et les centrant automatiquement
    autour de leur région d'intérêt, puis les redimensionnant et les enregistrant dans un dossier de sortie.
    
    :param folder_path: Chemin vers le dossier contenant les images.
    :param output_folder: Chemin vers le dossier où les images traitées seront enregistrées.
    :param canvas_width: Largeur du canevas pour les images de sortie.
    :param canvas_height: Hauteur du canevas pour les images de sortie.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            output_path = os.path.join(output_folder, f"{filename}")
            auto_crop_and_center_image(image_path, output_path, canvas_width, canvas_height)
            print(f"Processed {filename}")


def check_cropping_quality(image_path):
    """
    Vérifie si une image a été mal recadrée en déterminant s'il y a des lignes droites provenant d'un mauvais recadrage.

    :param image_path: Le chemin de l'image à vérifier.
    :return: Un booléen indiquant si l'image est mal recadrée.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Could not open or find the image {image_path}")
        return True 

    _, thresh = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)

    lines = cv2.HoughLinesP(thresh, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)

    if lines is None or len(lines) == 0:
        print(f"No straight lines found in image {image_path}. Assuming bad cropping.")
        return True

    return False

def check_folder_cropping_quality(folder_path):
    """
    Vérifie la qualité du recadrage de toutes les images dans un dossier et affiche les résultats.
    
    :param folder_path: Chemin vers le dossier contenant les images.
    """
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            if check_cropping_quality(image_path):
                print(f"Image {filename} has been badly cropped.")


train = "data/train"
test = "data/test"
val = "data/val"
images, labels = load_data(train)

# Charger les données de test
test_images, test_labels = load_data(test)

# Créer un modèle VGG-16 pré-entraîné (ne pas inclure la couche dense finale)
base_model = VGG16(include_top=False, input_shape=(224, 224, 3))

NUM_CLASSES = 1

model = Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(NUM_CLASSES, activation='sigmoid'))

# Figer les poids du VGG
model.layers[0].trainable = False

# Compiler le modèle
model.compile(
    loss='binary_crossentropy',
    optimizer=RMSprop(lr=1e-4),
    metrics=['accuracy']
)

# Afficher la structure du modèle
model.summary()

# Créer un générateur d'images pour la data augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.05,
    height_shift_range=0.05,
    rescale=1./255,
    shear_range=0.05,
    brightness_range=[0.1, 1.5],
    horizontal_flip=True,
    vertical_flip=True
)

# Ajuster le générateur aux données d'entraînement
datagen.fit(images)

# Entraîner le modèle avec l'augmentation de données
model.fit(datagen.flow(images, labels, batch_size=32),
          epochs=10,
          steps_per_epoch=len(images) // 32,
          validation_data=(test_images, test_labels))

# Évaluer le modèle sur les données de test
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')
