import os
import numpy as np
import cv2
from keras.models import load_model

# Charger le modèle pré-entraîné
model = load_model('model.h5')

# Charger les labels à partir du fichier 'labels.csv'
labelNames = open("labels.csv").read().strip().split("\n")[1:]
labelNames = [l.split(",")[1] for l in labelNames]

# Dossier contenant les images à tester
image_dir = 'Meta'  # Remplace par le chemin vers ton dossier d'images

# Fonction de prétraitement des images
def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def equalize(img):
    img = cv2.equalizeHist(img)
    return img

def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255.0  # Normalisation
    return img

# Fonction pour traiter les images et faire des prédictions
def process_images(image_dir):
    # Créer le dossier de sortie si tu veux enregistrer les résultats
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Parcourir toutes les images dans le dossier
    for filename in os.listdir(image_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):  # Vérifier les extensions des fichiers
            image_path = os.path.join(image_dir, filename)
            
            # Charger l'image
            img = cv2.imread(image_path)

            # Prétraiter l'image
            img_resized = cv2.resize(img, (32, 32))  # Redimensionner à 32x32 pixels
            img_processed = preprocessing(img_resized)  # Appliquer le prétraitement

            # Ajouter les dimensions batch et canal
            img_processed = np.expand_dims(img_processed, axis=-1)  # Ajouter une dimension pour le canal (grayscale)
            img_processed = np.expand_dims(img_processed, axis=0)  # Ajouter une dimension pour le batch

            # Faire une prédiction avec le modèle
            preds = model.predict(img_processed)
            predicted_class = np.argmax(preds, axis=1)[0]  # La classe prédite
            label = labelNames[predicted_class]  # Nom de la classe prédite

            # Afficher le résultat dans la console
            print(f"Image: {filename}, Predicted Class: {label}")

            # Afficher l'image avec le label prédit sur la fenêtre
            cv2.putText(img, label, (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow(f'Predicted: {label}', img)

            # Enregistrer l'image avec le label prédit si tu veux
            output_image_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_image_path, img)  # Enregistrer l'image dans le dossier 'output'

            # Attendre que l'utilisateur appuie sur une touche pour fermer la fenêtre d'affichage
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break  # Quitter si 'q' est pressé

    # Fermer toutes les fenêtres ouvertes
    cv2.destroyAllWindows()

# Lancer le traitement des images
process_images(image_dir)
