import numpy as np
import cv2
import tensorflow as tf
from keras.models import load_model
import csv
import time

# Fonction pour charger les labels depuis 'labels.csv'
def load_labels(label_file):
    label_dict = {}
    with open(label_file, mode='r') as infile:
        reader = csv.reader(infile)
        next(reader)  # Sauter l'en-tête
        for rows in reader:
            label_dict[int(rows[0])] = rows[1]
    return label_dict

# Fonction de prétraitement de l'image pour YOLO
def preprocess_image(frame):
    # Convertir l'image en niveaux de gris
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Redimensionner à la taille attendue par ton modèle (32x32)
    resized_frame = cv2.resize(gray_image, (32, 32))
    # Ajouter une dimension supplémentaire pour la profondeur (1 canal)
    input_image = np.expand_dims(resized_frame, axis=-1)
    # Ajouter la dimension de lot (None, 32, 32, 1)
    input_image = np.expand_dims(input_image, axis=0)
    input_image = input_image.astype('float32') / 255.0
    return input_image


# Fonction pour dessiner les boîtes de détection
def draw_boxes(frame, detections, label_dict, threshold=0.5):
    h, w, _ = frame.shape
    for detection in detections:
        # Ici on suppose que 'detection' est dans le format (x_center, y_center, width, height, score, class_id)
        if detection[4] > threshold:  # Si le score est supérieur au seuil
            x_center, y_center, width, height, score, class_id = detection
            # Calcul des coordonnées de la boîte (en pixels)
            x_min, y_min = int((x_center - width / 2) * w), int((y_center - height / 2) * h)
            x_max, y_max = int((x_center + width / 2) * w), int((y_center + height / 2) * h)
            
            # Dessiner la boîte
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            
            # Afficher la classe et la confiance
            label = label_dict.get(class_id, f"Class {class_id}")  # Récupérer le nom de la classe depuis le label_dict
            cv2.putText(frame, f'{label}: {score:.2f}', (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return frame

# Charger les labels depuis labels.csv
label_dict = load_labels('labels.csv')

# Charger le modèle YOLO (ton modèle .h5)
model = load_model('model.h5')

# Initialiser la webcam
cap = cv2.VideoCapture(0)
time.sleep(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Prétraiter l'image avant de la passer dans le modèle YOLO
    input_image = preprocess_image(frame)

    # Effectuer la prédiction avec le modèle
    predictions = model.predict(input_image)[0]  # Assurez-vous de récupérer la bonne dimension

    # Inspecter la forme de la prédiction
    print("Forme des prédictions:", predictions.shape)

    # Traitement des prédictions
    detections = []
    for i in range(predictions.shape[0]):
        for j in range(predictions.shape[1]):
            for b in range(predictions.shape[2]):
                # Récupérer les prédictions de la boîte (formées comme : [x, y, w, h, conf, class_id])
                box = predictions[i, j, b]  # Cette ligne suppose que les boîtes sont dans la troisième dimension

                # Vérifier la forme de 'box'
                if box.shape[0] >= 5:  # Vérifier que le tableau a au moins 5 éléments (confiance + coordonnées)
                    confidence = box[4]
                    if confidence > 0.5:  # Seuil de confiance
                        class_id = np.argmax(box[5:])  # L'ID de la classe avec la confiance la plus élevée
                        detections.append((box[0], box[1], box[2], box[3], confidence, class_id))

    # Dessiner les boîtes sur la frame
    for detection in detections:
        x, y, w, h, confidence, class_id = detection
        label = label_dict.get(class_id, 'Unknown')
        cv2.rectangle(frame, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), (0, 255, 0), 2)
        cv2.putText(frame, f"{label}: {confidence:.2f}", (int(x - w / 2), int(y - h / 2) - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Afficher la frame avec les détections
    cv2.imshow("Detection", frame)

    # Quitter avec la touche 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Nettoyage
cap.release()
cv2.destroyAllWindows()