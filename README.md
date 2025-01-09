

# Traffic Sign Recognition with TensorFlow

## Overview
Ce projet implémente un réseau de neurones en utilisant TensorFlow pour classifier des images de panneaux de signalisation à partir du **German Traffic Sign Recognition Benchmark (GTSRB)**. Le modèle identifie avec précision différents types de panneaux de signalisation, tels que les panneaux d'arrêt, les limites de vitesse, et les panneaux de céder le passage, entre autres.

## Getting Started

Pour commencer, assurez-vous d'avoir **Python 3.10.8** installé, car cette version est recommandée pour la compatibilité avec TensorFlow.

### Prérequis

- **Python** (version 3.10.8 recommandée)
- **Git** installé pour cloner le dépôt.

### 1. Cloner le dépôt

```bash
git clone https://github.com/Helmisoudana/TrafficSignDetection
cd TrafficSignDetection
```

### 2. Télécharger le dataset

Téléchargez le dataset du projet à partir du lien suivant et placez-le dans ce répertoire :

[Dataset GTSRB - Google Drive](https://drive.google.com/file/d/1Tzw4hHHRIhkJpCeFye5kafD_Go7A4vNv/view)

### 3. Créer un environnement virtuel

Créez un environnement virtuel pour isoler les dépendances du projet :

Sous Windows :
```bash
python -m venv env
env\Scripts\activate
```

Sous MacOS/Linux :
```bash
python3 -m venv env
source env/bin/activate
```

### 4. Installer les dépendances

Dans le répertoire du projet, exécutez la commande suivante pour installer toutes les dépendances nécessaires (comme OpenCV-Python, scikit-learn et TensorFlow) :

```bash
pip install -r requirements.txt
```

### 5. Exécuter le programme

Une fois les dépendances installées, vous pouvez lancer les scripts suivants pour entraîner le modèle, prédire les panneaux de signalisation, ou utiliser la caméra pour la détection :

- Entraîner le modèle :
  ```bash
  python train.py
  ```

- Faire des prédictions :
  ```bash
  python predict.py
  ```

- Utiliser la webcam pour la détection :
  ```bash
  python cvcam.py
  ```

## Implementation Details

### `train.py`

Le cœur du projet réside dans `train.py`, où nous implémentons deux fonctions principales :

1. **load_data(data_dir)** : Cette fonction charge les données d'image et les étiquettes correspondantes à partir du répertoire spécifié (`data_dir`). Chaque image est redimensionnée à une taille standard (IMG_WIDTH x IMG_HEIGHT) en utilisant OpenCV-Python (`cv2`) et convertie en tableau numpy. Elle renvoie deux listes : `images`, contenant les tableaux d'images, et `labels`, contenant la catégorie de chaque image.

2. **get_model()** : Cette fonction construit et compile un modèle de réseau de neurones en utilisant l'API Keras de TensorFlow. L'architecture du modèle est personnalisable, permettant d'expérimenter avec différentes configurations de couches convolutionnelles, de pooling, de couches cachées et de taux de dropout pour optimiser l'exactitude du modèle.

### Training and Evaluation

Une fois les données chargées et le modèle construit, `train.py` entraîne le modèle sur l'ensemble d'entraînement et évalue ses performances sur l'ensemble de test. Les progrès de l'entraînement, y compris les métriques de perte et de précision pour chaque époque, sont affichés dans la console.

## Experimentation

Tout au long du projet, vous êtes encouragé à expérimenter avec différentes architectures de modèles et hyperparamètres. Vous pouvez modifier la fonction `get_model()` pour explorer :

- Différents nombres de couches convolutionnelles et de pooling.
- La taille et le nombre de filtres dans les couches convolutionnelles.
- Diverses configurations de couches cachées et de taux de dropout.

## Contribuer

Si vous souhaitez contribuer à ce projet, merci de faire un fork du dépôt et de soumettre une pull request.

---

En utilisant ce guide, les utilisateurs pourront facilement configurer l'environnement virtuel, installer les dépendances et commencer à travailler avec ton projet de reconnaissance de panneaux de signalisation.
