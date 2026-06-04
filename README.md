# TP de CNN from scratch vs Transfer Learning : Classification Chats/Chiens

**Auteur** : Eudoxie ABOUTA  
**Cours** : Master Intelligence Artificielle au DIT  

---

## Objectif

Comparer un **CNN entraine from scratch** et un modele en **transfert learning (MobileNetV2)** sur le dataset Cats vs Dogs (22 500 images train / 2 500 images test).

---

## Environnement

```bash
pip install -r requirements.txt
```

Notebook concu pour **Google Colab avec GPU** (teste sur NVIDIA A100-SXM4-80GB).

---

## Organisation des donnees

Le dataset n'est **pas** inclus dans ce depot.

1. Telecharger `Cat_Dog_data.zip` depuis le Drive du cours
2. Placer dans `My Drive/Deep_learning/`
3. Le notebook decompresse automatiquement au premier lancement

Structure attendue :

```
Cat_Dog_data/
  train/
    cat/   (11 250 images)
    dog/   (11 250 images)
  test/
    cat/   (1 250 images)
    dog/   (1 250 images)
```

---

## Structure du depot

```
tp_deeplearning_eudoxie/
|-- Eudoxie_ABOUTA_Loading_Image_Data_TP_FINAL.ipynb   # Notebook principal
|-- README.md
|-- requirements.txt                                    # Dependances Python
|-- .gitignore
```

Fichiers **non pousses** sur GitHub :

```
Cat_Dog_data/        # Dataset brut
*.pth / *.pt         # Checkpoints des modeles
runs/                # Logs TensorBoard
__pycache__/
.ipynb_checkpoints/
```

---

## Entrainement

### Experience A - CNN from scratch

| Config | lr | Epochs | Scheduler |
|---|---|---|---|
| SGD + momentum=0.9 | 1e-2 | 20 | StepLR(step=7) |
| Adam | 1e-3 | 20 | CosineAnnealingLR |

Architecture : 3 blocs (Conv + BN + ReLU + MaxPool) + FC(512) + Dropout(0.5) + FC(2)

### Experience B - Transfer Learning (MobileNetV2)

| Phase | Config | lr | Epochs |
|---|---|---|---|
| Feature extraction | Adam, couches gelees | 1e-3 | 10 |
| Fine-tuning | SGD, tout degele | 1e-4 | 10 |

---

## Evaluation et rechargement du modele

```python
from torchvision.models import mobilenet_v2
import torch, torch.nn as nn

model = mobilenet_v2(weights=None)
model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(1280, 2))
model.load_state_dict(torch.load('best_tl_sgd.pth', map_location='cpu'))
model.eval()
```

---

## Resultats

| Modele | Test Acc | Precision | Recall |
|---|---|---|---|
| CNN Scratch - SGD | 50.0% | 50.0% | 100% |
| CNN Scratch - Adam | 80.6% | 85.9% | 98.1% |
| TL MobileNetV2 - Adam (feat. ext.) | 97.8% | 98.4% | 97.2% |
| **TL MobileNetV2 - SGD (fine-tune)** | **98.7%** | **98.9%** | **98.6%** |

**Rapport final sur 2 500 images (best_tl_sgd.pth) :**

```
              precision    recall  f1-score
         cat       0.99      0.99      0.99
         dog       0.99      0.99      0.99
    accuracy                           0.99
```

32 erreurs sur 2 500 images (taux d'erreur = 1.28%)

---

## Analyse

Le **CNN from scratch avec SGD** echoue completement : il reste a 50% d'accuracy (equivalent au hasard) sur 20 epoques. Le lr=0.01 est trop eleve, les gradients oscillent et le modele reste bloque dans un minimum trivial. **Adam** resout ce probleme grace a l'adaptation du lr par parametre et converge jusqu'a 80.6% en 20 epoques, montrant qu'un CNN from scratch reste possible mais lent sur un dataset de 22 500 images.

Le **transfer learning (feature extraction)** obtient 97.0% des la premiere epoque avec seulement 2 562 parametres entraines. MobileNetV2 pre-entraine sur ImageNet possede deja des filtres pour les textures de fourrure, les contours d'oreilles et de museaux. Le **fine-tuning SGD** pousse encore jusqu'a 98.7% en adaptant finement l'ensemble du reseau avec un tres faible lr (1e-4) qui preserve les features acquises.

L'ecart de 18 points entre scratch Adam (80.6%) et TL fine-tune (98.7%) illustre la puissance du transfer learning sur des datasets de taille moderee. Les 32 erreurs residuelles concernent principalement des chats a museau allonge confondus avec des chiens. Pour franchir 99% : augmentation avancee (Mixup/CutMix), split validation + early stopping, ou un backbone plus puissant (EfficientNet-B4).

---

## Limites et pistes d'amelioration

- Split train/val/test explicite + early stopping
- Augmentation avancee : Mixup, CutMix, RandAugment
- Tester EfficientNet-B4 ou ResNet-50
- TensorBoard pour le suivi en temps reel

---

## Reproductibilite

Seed fixe a **42** pour `random`, `numpy`, `torch` et `torch.cuda`.