# Digital Document Tampering Detection

Pixel-level localization of digitally tampered regions in identity documents and passports, using a U-Net-style convolutional encoder–decoder trained on synthetically forged documents built from the MIDV-2020 dataset.

The framing choice is what makes this project interesting. Most document forensics work asks "is this document fake?" — a binary classification. This asks **"which pixels were changed?"**, which is the question an actual document examiner needs answered. A binary verdict tells a border officer nothing actionable; a mask pointing at the date-of-birth field does.

> Neither the MIDV-2020 source data nor the trained weights are included in this repository.

---

## The forgery generation problem

Supervised tampering localization needs pixel-accurate ground-truth masks, and no public dataset of real forged IDs with annotated tamper regions exists at meaningful scale. So the forgeries have to be manufactured.

**`ForgingAnnotation.ipynb`** builds them from MIDV-2020's VIA-format JSON annotations, which carry bounding boxes for each document field:

| Field group | Region indices |
|---|---|
| Text attributes (name, surname, etc.) | 2, 3, 5, 6, 7, 8 |
| Date attributes | 9, 10, 11 |

For each document, the generator picks two distinct attribute regions at random, crops both, and swaps them — producing a document where two fields have exchanged content while everything else is pixel-identical to the original. A binary mask marking both swapped regions is written alongside. Up to 30 distinct attribute pairs are generated per source document, with already-used pairs tracked to avoid duplicates.

Swapping fields *within* a document rather than pasting in external content is a deliberate choice: it keeps font, resolution, lighting, and JPEG history consistent, so the model can't cheat by detecting a mismatch in low-level image statistics. It has to learn something about content plausibility.

**Output:** 21,698 tampered images with matching masks, from which **6,001** were randomly sampled for training.

```
AnnotatedData/
├── TrainingImages/img/     # 6,001 tampered documents
└── TrainingMasks/img/      # 6,001 binary masks
```

Image and mask filenames are kept identical (`NNNNN_tampered.jpg` in both directories) — this matters for how the training pipeline pairs them.

---

## Architecture

**`Modeling.ipynb`** implements a U-Net-style encoder–decoder from scratch, no pretrained backbone.

```
Input 256×256×3
  ├─ Conv(16) ×2  ──────────────────────────────┐  skip
  │  MaxPool → 128×128                          │
  ├─ Conv(32) ×2  ────────────────────────────┐ │  skip
  │  MaxPool → 64×64                          │ │
  ├─ Conv(64) ×2  ──────────────────────────┐ │ │  skip
  │  MaxPool → 32×32                        │ │ │
  ├─ Conv(128) ×2  ───────────────────────┐ │ │ │  skip
  │  MaxPool → 16×16                      │ │ │ │
  ├─ Conv(256) ×2  ─────────────────────┐ │ │ │ │  skip
  │  MaxPool → 8×8                      │ │ │ │ │
  └─ Center: Conv(256)                  │ │ │ │ │
     UpSample → 16×16 ──── Concatenate ─┘ │ │ │ │
     Conv(256) ×2                         │ │ │ │
     UpSample → 32×32 ──── Concatenate ───┘ │ │ │
     Conv(128) ×2                           │ │ │
     UpSample → 64×64 ──── Concatenate ─────┘ │ │
     Conv(64) ×2                              │ │
     UpSample → 128×128 ─── Concatenate ──────┘ │
     Conv(32) ×2                                │
     UpSample → 256×256 ─── Concatenate ────────┘
     Conv(16) ×2
     Conv(1, 1×1, sigmoid)
Output 256×256×1
```

| Parameter | Value |
|---|---|
| Input | 256 × 256 × 3 |
| Output | 256 × 256 × 1 |
| Encoder channels | 16 → 32 → 64 → 128 → 256 |
| Skip connections | 5 |
| Trainable parameters | 4,322,689 |
| Output activation | Sigmoid |

All convolutions are 3×3 with `same` padding; upsampling is nearest-neighbour rather than transposed convolution, which avoids checkerboard artifacts at the cost of some expressiveness.

The skip connections are the critical piece for this task. By the 8×8 bottleneck, a tampered text field has been compressed to a handful of spatial positions — nowhere near enough to draw a mask boundary. Concatenating the full-resolution encoder features back in on the way up restores the spatial precision the localization needs.

### Loss

Binary cross-entropy plus Dice loss:

```python
dice_coeff = (2·|y_true ∩ y_pred| + 1) / (|y_true| + |y_pred| + 1)
dice_loss  = 1 − dice_coeff
bce_dice_loss = binary_crossentropy + dice_loss
```

The combination matters because tampered regions occupy a small fraction of each document. BCE alone would be minimized well by a model that predicts "background" everywhere — that's over 95% correct on a pixel count. Dice measures overlap between predicted and true regions, so it stays sensitive to the minority class. The `smooth = 1.` term prevents division by zero on images with an empty mask.

---

## Training

| Setting | Value |
|---|---|
| Train / validation | 4,801 / 1,200 (80/20 split) |
| Batch size | 2 |
| Steps per epoch | 2,400 |
| Optimizer | RMSprop, `learning_rate=0.0015` |
| Epochs | 70 (all completed) |
| Checkpoint | best `val_loss` |
| Time per epoch | ~800 s (~15.5 hours total, CPU) |

Images and masks are loaded through **separate** `ImageDataGenerator` flows — RGB for images, grayscale for masks — and paired with `zip()`. Both use `seed=33` so the shuffling stays aligned, which works here because the two directories contain identically-named files that sort into the same order.

### Results

| Metric | Train | Validation |
|---|---|---|
| Loss | 0.4717 | **0.6768** |
| Dice coefficient | 0.5514 | **0.3594** |

Best checkpoint at **epoch 50**. No subsequent epoch improved on it across the remaining 20.

### Reading these numbers honestly

A validation Dice of 0.36 means predicted and true tamper regions overlap by about a third. The model has learned something — random output would score far lower — but this is not a working detector.

Two things are visible in the training log. First, **the gap doesn't close**: training Dice climbs steadily from 0.53 to 0.56 while validation oscillates between 0.30 and 0.36 with no trend. Second, **validation loss is almost entirely Dice loss**. With `val_dice ≈ 0.36`, the Dice component alone is ≈ 0.64, leaving only ≈ 0.04 for BCE. The model easily gets the background right and struggles specifically at the thing the task is about — the boundaries of the tampered regions.

The first thing I'd investigate is the mask format; see *Known issues*.

---

## Inference

```python
img = cv2.imread('document.jpg')
img = cv2.resize(img, (256, 256))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
x   = np.expand_dims(img.astype("float32") / 255.0, axis=0)
mask = model.predict(x)
binary = (mask[0, :, :, 0] >= threshold) * 255
```

---

## Running it

```bash
git clone https://github.com/RovshanBayramRB/Digital-Document-Tampering-Detection.git
cd Digital-Document-Tampering-Detection
pip install tensorflow opencv-python numpy matplotlib jupyter
```

1. Obtain [MIDV-2020](http://l3i-share.univ-lr.fr/MIDV2020/midv2020.html) — document images plus VIA JSON annotations.
2. Open `ForgingAnnotation.ipynb`, set the dataset paths, uncomment the generation cells, and run to produce tampered images and masks.
3. Open `Modeling.ipynb`, point it at the generated `TrainingImages/` and `TrainingMasks/` directories, and train.

Expect roughly 13 minutes per epoch on CPU. A GPU runtime is strongly recommended.

---

## Repository structure

```
.
├── ForgingAnnotation.ipynb   # Synthetic forgery generation and mask annotation
├── Modeling.ipynb            # Architecture, training, inference
└── README.md
```
