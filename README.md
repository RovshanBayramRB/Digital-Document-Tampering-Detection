Digital Document Tampering Detection

A computer vision project for localizing digitally tampered regions in identity documents and passports using pixel-level image segmentation.

The project uses the MIDV-2020 document dataset to construct a tampered-document dataset and trains a convolutional encoder-decoder network to generate segmentation masks identifying potentially manipulated regions.

Overview

Digital manipulation of identity documents can involve changing specific attributes such as names, dates, or other personal information while leaving the rest of the document visually unchanged.

Rather than treating tampering detection as a simple binary classification problem, this project approaches it as an image segmentation task. The model predicts a pixel-level mask indicating the regions that are likely to have been modified.

The project consists of two primary stages:

1. Forgery generation and annotation — document metadata is used to identify attribute regions and generate annotated tampered images.
2. Tampering localization — a CNN-based encoder-decoder model is trained using the generated images and segmentation masks.

Dataset

The project is based on the MIDV-2020 dataset, which contains identity documents and passports from multiple countries.

The original project description reports:

- 1,000 document images
- Documents from 10 countries
- JSON metadata containing coordinates for document attributes such as names, surnames, dates, and other fields

The original dataset is not included in this repository.

Data Preparation

The "ForgingAnnotation.ipynb" notebook is used for preparing the training data.

The workflow uses the coordinate information stored in the dataset's JSON metadata to identify document attributes. OpenCV and Python filesystem utilities are then used to create manipulated document images and corresponding annotation masks.

The resulting dataset is organized into separate image and mask directories:

AnnotatedData/
├── TrainingImages/
│   └── img/
└── TrainingMasks/
    └── img/

The modeling notebook subsequently pairs each tampered image with its corresponding segmentation mask.

Model Architecture

The project implements a custom convolutional encoder-decoder segmentation architecture with skip connections.

The encoder progressively reduces the spatial resolution while increasing the number of feature channels:

256 × 256 × 3
       │
       ▼
   Conv Blocks
       │
   Max Pooling
       │
       ▼
   128 × 128
       │
   Max Pooling
       │
       ▼
    64 × 64
       │
   Max Pooling
       │
       ▼
    32 × 32
       │
   Max Pooling
       │
       ▼
    16 × 16
       │
   Max Pooling
       │
       ▼
     8 × 8
       │
     Center
       │
       ▼
  Upsampling
       │
  Skip Connections
       │
       ▼
256 × 256 × 1

The decoder progressively restores the original spatial resolution. Feature maps from the encoder are concatenated with corresponding decoder representations through skip connections, allowing the network to retain spatial information needed for localization.

The final layer uses a "1×1" convolution with a sigmoid activation to produce a single-channel segmentation mask.

Model configuration

Parameter| Value
Input size| 256 × 256 × 3
Output| 256 × 256 × 1
Architecture| CNN encoder-decoder
Skip connections| Yes
Trainable parameters| 4,322,689
Output activation| Sigmoid

The architecture and parameter count are taken directly from the modeling notebook.

Training

Images and masks are resized to 256 × 256 and normalized by dividing pixel values by 255.

The training pipeline uses:

- Batch size: "2"
- Validation split: "20%"
- Training images: "4,801"
- Validation images: "1,200"
- Optimizer: RMSprop
- Learning rate: "0.0015"
- Training duration: up to 70 epochs

The notebook uses Keras "ImageDataGenerator" instances to load the document images and grayscale segmentation masks.

Loss Function

The model combines Binary Cross-Entropy and Dice loss:

BCE + Dice Loss

Dice loss is defined as:

Dice Loss = 1 - Dice Coefficient

This combination is suitable for segmentation because it considers both pixel-wise classification and overlap between the predicted and ground-truth regions.

Checkpointing

The best model checkpoint is selected according to validation loss:

ModelCheckpoint(
    monitor="val_loss",
    save_best_only=True,
    mode="min"
)

The trained model is saved as "best_model_2.h5" during the notebook workflow.

Results

The training notebook records the following best validation result:

Metric| Best recorded value
Validation loss| ~0.6768
Validation Dice coefficient| ~0.3594
Epoch| 50

The best validation loss was recorded at epoch 50, after which subsequent epochs did not improve the checkpoint according to the monitored validation loss.

These results should be interpreted as the results of the original experiment rather than as a benchmark against current state-of-the-art document tampering detection methods.

Inference

The modeling notebook includes a testing workflow that loads a trained Keras model and performs inference on an individual document image.

The image is:

1. Loaded with OpenCV.
2. Resized to 256 × 256.
3. Converted to floating-point representation.
4. Normalized to "[0, 1]".
5. Expanded with a batch dimension.
6. Passed through the segmentation model.

The model outputs a pixel-level prediction mask representing the detected tampered regions.

Repository Structure

Digital-Document-Tampering-Detection/
│
├── ForgingAnnotation.ipynb
├── Modeling.ipynb
└── README.md

Notebooks

"ForgingAnnotation.ipynb"

Data preparation and annotation workflow used to generate tampered document images and corresponding masks.

"Modeling.ipynb"

Contains:

- Dataset loading
- Image/mask preprocessing
- Model architecture
- Custom loss functions
- Model training
- Checkpointing
- Model testing and inference

The repository currently consists primarily of these two Jupyter notebooks.

Technologies

- Python
- TensorFlow
- Keras
- OpenCV
- NumPy
- Matplotlib
- Jupyter Notebook / Google Colab

Getting Started

1. Clone the repository

git clone https://github.com/RovshanBayramRB/Digital-Document-Tampering-Detection.git
cd Digital-Document-Tampering-Detection

2. Install dependencies

The original project was developed using TensorFlow/Keras and supporting Python libraries.

For example:

pip install tensorflow opencv-python numpy matplotlib jupyter

3. Obtain the dataset

Download the required MIDV-2020 dataset separately and prepare the document images and JSON metadata.

The dataset is not distributed with this repository.

4. Generate annotated data

Open:

ForgingAnnotation.ipynb

Configure the dataset paths and execute the notebook to generate the tampered images and corresponding masks.

5. Train the model

Open:

Modeling.ipynb

Configure the paths to the generated training data and execute the notebook.

Limitations

This repository represents an experimental research project and is primarily implemented as Jupyter notebooks.

Some aspects of the original implementation are environment-specific, including Google Colab/Google Drive paths and locally stored model files. Therefore, additional configuration may be required to reproduce the original experiment.

The repository does not currently include:

- A standalone inference script
- A packaged Python application
- Automated tests
- Dependency lock files
- Trained model weights
- The original dataset

Future Improvements

Potential directions for extending the project include:

- Refactoring the notebook workflow into reusable Python modules
- Adding a standalone inference CLI
- Providing reproducible dependency management
- Adding automated evaluation metrics such as IoU and F1-score
- Experimenting with modern segmentation architectures
- Improving the training/validation data pipeline
- Adding visualization utilities for predicted tampering masks
- Including representative input/output examples
- Providing a lightweight demo application

Author

Rovshan Bayram

AI Engineer | Data Scientist

"GitHub" (https://github.com/RovshanBayramRB)