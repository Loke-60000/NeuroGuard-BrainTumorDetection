import matplotlib.pyplot as plt
import numpy as np

def plot_images(X, y, n):

    """
    Affiche les images à partir des données X et les labels y.
    
    :param X: Les données d'image.
    :param y: Les labels correspondantes aux images.
    :param n: Le nombre total d'images à afficher.
    """
    plt.figure(figsize=(10, 10))
    yes_indices = [i for i, label in enumerate(y) if label == 1]
    no_indices = [i for i, label in enumerate(y) if label == 0]
    
    total_yes = len(yes_indices)
    total_no = len(no_indices)

    # Déterminez le nombre d'images "oui" et "non" à afficher
    num_yes = min(n // 2, total_yes)
    num_no = min(n - num_yes, total_no)
    
    for i in range(n):
        plt.subplot(2, num_yes + num_no, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        if i < num_yes:
            plt.imshow(X[yes_indices[i]], cmap=plt.cm.binary)
            plt.xlabel("yes")
        else:
            plt.imshow(X[no_indices[i - num_yes]], cmap=plt.cm.binary)
            plt.xlabel("no")
    plt.show()

   


    