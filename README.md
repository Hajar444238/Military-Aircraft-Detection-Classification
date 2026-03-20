# 🛩️ Détection et Classification d'Aéronefs Militaires

> Pipeline complet de Computer Vision pour détecter et classifier automatiquement les aéronefs militaires à partir d'images aériennes, en utilisant YOLOv8 pour la détection et plusieurs approches CNN pour la classification.


---

## 📋 Description du projet

Dans un contexte de surveillance aérienne en zone de conflit, la détection manuelle des aéronefs est lente, coûteuse et sujette à des erreurs humaines. Ce projet répond à la problématique suivante :

> *Comment concevoir un système capable de détecter automatiquement les aéronefs militaires dans des images aériennes et de classifier leur type avec précision, malgré des conditions complexes telles que le bruit, les variations d'éclairage, la distance et les angles de vue ?*

La solution proposée combine :
- **YOLOv8s** pour la détection et la localisation (bounding box)
- **Random Forest** comme approche classique de classification (baseline)
- **CNN ResNet50** comme approche Deep Learning de classification
- **EfficientNetV2** comme approche Deep Learning avancée de classification

---

## 📊 Dataset

**Military Aircraft Detection Dataset** — disponible sur [Kaggle](https://www.kaggle.com/datasets/a2015003713/militaryaircraftdetectiondataset)

| Statistique | Valeur |
|---|---|
| Nombre d'images | 23 126 |
| Nombre d'annotations | 41 441 |
| Nombre de classes | 101 |
| Moyenne aéronefs/image | 1.79 |

### Répartition train / val / test

| Split | Annotations | Pourcentage |
|---|---|---|
| Train | 31 127 | 75% |
| Validation | 7 545 | 18% |
| Test | 2 769 | 7% |

### Structure du dataset

```
militaryaircraftdetectiondataset/
├── dataset/              ← Images brutes .jpg + CSV par image → DÉTECTION
├── crop/                 ← Images découpées par classe (101 dossiers) → CLASSIFICATION
├── annotated_samples/    ← Images avec bbox dessinées → vérification visuelle
└── labels_with_split.csv ← 41 441 annotations : filename, width, height, class,
                             xmin, ymin, xmax, ymax, split
```

### Classes d'aéronefs (101 classes)

`A10` `A400M` `AG600` `AH64` `AV8B` `B1` `B2` `B21` `B52` `C130` `C17` `C5`
`EF2000` `F14` `F15` `F16` `F18` `F22` `F35` `F4` `J20` `JAS39` `Ka52`
`Mi24` `Mi28` `Mig29` `Mirage2000` `MQ9` `Rafale` `ResNet50` `SR71` `Su57`
`Tu95` `Tu160` `U2` `UH60` `Vulcan` `X32` `YF23` `Z19` *(et 61 autres)*

---

## 🔧 Pipeline du projet

```
Dataset (images + CSV)
        ↓
Nettoyage des annotations (59 bbox invalides supprimées)
        ↓
Encodage des classes (A10=0, F16=31, Rafale=69 ... Z19=100)
        ↓
Conversion format Pascal VOC → YOLO (coordonnées normalisées)
        ↓
Organisation train / val / test (23 126 paires image + label)
        ↓
Création aircraft.yaml (configuration YOLOv8)
        ↓
        ├── Entraînement YOLOv8s (50 epochs, GPU Tesla T4, ~5h)
        │           ↓
        │       best.pt (21.5 MB)
        │           ↓
        │   Évaluation sur split test (1 544 images)
        │
        └── Entraînement Classification
                    ├── Random Forest (HOG features, 1816 dim.)
                    ├── CNN ResNet50 (Transfer Learning, 20 epochs)
                    └── EfficientNetV2 (Transfer Learning)
```

---

## 🎯 Résultats

### Détection — YOLOv8s

| Métrique | Valeur | Signification |
|---|---|---|
| **mAP@0.5** | **0.743** | Précision moyenne sur 101 classes |
| Précision | 0.821 | 82% des détections sont correctes |
| Rappel | 0.698 | 70% des aéronefs présents sont trouvés |
| Vitesse | 8.6 ms/image | Compatible temps réel (~116 img/sec) |

**Top 5 meilleures classes :**

| Classe | mAP@0.5 |
|---|---|
| X32 | 0.932 |
| Tu95 | 0.924 |
| US2 | 0.849 |
| Vulcan | 0.815 |
| Tu160 | 0.796 |

### Classification

| Modèle | Top-1 Accuracy | Top-5 Accuracy | Vitesse |
|---|---|---|---|
| Random Forest | 16.29% | 36.15% | 6837 FPS |
| CNN ResNet50 | 70.49% | 89.27% | 122.6 FPS |
| EfficientNetV2 | — | — | — |

---

## 🏗️ Architecture des modèles

### YOLOv8s — Détection

```
Image (640×640)
    ↓
Backbone CSPDarknet  ← extraction des caractéristiques
    ↓
Neck PAN-FPN         ← fusion multi-échelle
    ↓
Head                 ← prédiction bbox + classe
    ↓
Output : bbox (cx, cy, bw, bh) + classe + confiance
```

**Paramètres d'entraînement :**
```
model   : yolov8s.pt (pré-entraîné COCO)
epochs  : 50
imgsz   : 640
batch   : 16
device  : GPU Tesla T4
durée   : ~5 heures
params  : 11 164 671
```

### CNN ResNet50 — Classification

```
Crop image (224×224)
    ↓
Conv Blocks + Residual Connections
    ↓
Global Average Pooling
    ↓
FC Layer (101 classes)
    ↓
Softmax → classe prédite
```

---

## 📁 Structure du repository

```
military-aircraft-detection-classification/
├── README.md                          ← ce fichier
├── requirements.txt                   ← dépendances Python
├── notebook.ipynb                     ← notebook principal (Kaggle)
└── presentation/
    └── presentation_final.pptx        ← présentation du projet
```

---

## 🚀 Comment reproduire les résultats

### Option 1 — Sur Kaggle (recommandé)

1. Allez sur [Kaggle](https://www.kaggle.com)
2. Créez un nouveau notebook
3. Ajoutez le dataset : **Military Aircraft Detection Dataset**
4. Activez le GPU : `Settings → Accelerator → GPU T4`
5. Copiez le contenu de `notebook.ipynb`
6. Cliquez **Run All**

### Option 2 — En local

```bash
# Cloner le repository
git clone https://github.com/votre_username/military-aircraft-detection-classification
cd military-aircraft-detection-classification

# Installer les dépendances
pip install -r requirements.txt

# Télécharger le dataset (nécessite un compte Kaggle)
kaggle datasets download -d a2015003713/militaryaircraftdetectiondataset
unzip militaryaircraftdetectiondataset.zip

# Lancer le notebook
jupyter notebook notebook.ipynb
```

---

## 📦 Dépendances

```
ultralytics>=8.0.0
torch>=2.0.0
torchvision>=0.15.0
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
scikit-learn>=1.2.0
opencv-python>=4.7.0
seaborn>=0.12.0
Pillow>=9.4.0
```

---

## 📈 Visualisations générées

Le notebook génère automatiquement les visualisations suivantes :

- Distribution des annotations par classe
- Répartition train / val / test
- Courbes de Loss pendant l'entraînement
- Courbes mAP, Précision et Rappel par epoch
- mAP@0.5 par classe (top 20)
- Images de détection avec bounding boxes
- Matrice de confusion (classification)
- Comparaison Random Forest vs CNN ResNet50

---

## 🔍 Exemples de détections

| Image | Classe détectée | Confiance |
|---|---|---|
| Base navale japonaise | US2 | 94% |
| Base militaire | Mirage2000 | 62% |
| Vol en altitude | An22 | 91% |
| Formation | F16 × 5 | ~80% |

---

## ⚠️ Limitations

- Les classes sous-représentées (WZ9=15 images, MQ20=11 images) obtiennent des scores mAP faibles
- Le déséquilibre entre les classes impacte les performances globales
- Le Random Forest atteint ses limites sur un problème à 101 classes complexes

---

## 🔮 Perspectives

- Tester YOLOv8l ou YOLOv8x pour de meilleures performances de détection
- Comparer avec Faster R-CNN (approche two-stage)
- Appliquer l'augmentation de données sur les classes sous-représentées
- Déploiement temps réel sur flux vidéo aérien

---

## 📚 Références

- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com)
- [Military Aircraft Detection Dataset — Kaggle](https://www.kaggle.com/datasets/a2015003713/militaryaircraftdetectiondataset)
- [ResNet50 — PyTorch](https://pytorch.org/vision/stable/models/resnet.html)
- [EfficientNet — PyTorch](https://pytorch.org/vision/stable/models/efficientnet.html)

---

## 📄 Licence

Ce projet est réalisé dans le cadre académique du cours de Traitement d'Images.
