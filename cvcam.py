import cv2
import numpy as np
import tensorflow as tf

# Charger le modèle YOLO .h5
model = tf.keras.models.load_model('model.h5')

# Démarrer la capture vidéo
cap = cv2.VideoCapture(0)  # 0 pour la caméra par défaut

# Fonction de prétraitement de l'image
# Fonction de prétraitement de l'image (convertir en niveaux de gris)
def preprocess_image(frame, target_size=(32, 32)):
    # Convertir l'image de BGR (OpenCV) en niveaux de gris
    image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Redimensionner l'image à la taille d'entrée du modèle
    image_resized = cv2.resize(image_gray, target_size)
    # Ajouter une dimension supplémentaire pour le canal (grayscale)
    image_resized = np.expand_dims(image_resized, axis=-1)  # Pour avoir la forme (32, 32, 1)
    # Normaliser l'image (diviser par 255)
    image_normalized = np.expand_dims(image_resized, axis=0).astype('float32') / 255.0
    return image_normalized


# Fonction de détection
def detect_objects(frame, model):
    input_image = preprocess_image(frame)
    detections = model.predict(input_image)
    return detections

# Fonction pour dessiner les boîtes de détection
def draw_boxes(frame, detections, threshold=0.5):
    h, w, _ = frame.shape
    for detection in detections[0]:  # Traitement pour une seule image
        score = detection[4]  # Le score de confiance pour cette boîte
        if score > threshold:  # Seulement afficher les objets avec un score > seuil
            # Extraire les coordonnées de la boîte (x, y, largeur, hauteur)
            x, y, w_box, h_box = detection[0:4]  # Ajuster selon ton modèle
            x_min, y_min, x_max, y_max = int(x * w), int(y * h), int((x + w_box) * w), int((y + h_box) * h)

            # Dessiner la boîte sur l'image
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(frame, f'Confidence: {score:.2f}', (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return frame

# Boucle de traitement en temps réel
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Détecter les objets
    detections = detect_objects(frame, model)

    # Dessiner les boîtes
    frame_with_boxes = draw_boxes(frame, detections)

    # Afficher la frame avec les boîtes de détection
    cv2.imshow('Real-Time Detection', frame_with_boxes)

    # Quitter avec 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer les ressources
cap.release()
cv2.destroyAllWindows()
