# 🧴 Skin Disease Classification System

> An advanced deep learning system for automated skin disease detection and classification using EfficientNetB0 and Flask web interface.

## 📋 Overview

This project implements a complete end-to-end skin disease classification system capable of identifying 22 different skin conditions from images. Built with TensorFlow/Keras and featuring a user-friendly Flask web application, it achieves high accuracy through transfer learning with EfficientNetB0 architecture, data augmentation, and fine-tuning strategies.

## ✨ Key Features

- **22 Disease Classes**: Comprehensive classification across diverse dermatological conditions
- **EfficientNetB0 Architecture**: State-of-the-art convolutional neural network optimized for accuracy and efficiency
- **Transfer Learning**: Pre-trained on ImageNet with custom fine-tuning for medical images
- **Advanced Training Pipeline**:
  - Data augmentation (rotation, flip, zoom, translation, contrast)
  - Class weight balancing for imbalanced datasets
  - Label smoothing for better generalization
  - Early stopping and learning rate scheduling
  - TensorBoard integration for training visualization
- **Web Interface**: Interactive Flask application for real-time predictions
- **Top-K Predictions**: Displays top 5 most likely diagnoses with confidence scores
- **Fast Inference**: Optimized prediction pipeline with preprocessing

## 🏥 Supported Skin Conditions

The model can classify the following 22 skin diseases:

<table>
<tr>
<td>

- Acne
- Actinic Keratosis
- Benign Tumors
- Bullous
- Candidiasis
- Drug Eruption
- Eczema
- Infestations/Bites

</td>
<td>

- Lichen
- Lupus
- Moles
- Psoriasis
- Rosacea
- Seborrheic Keratoses
- Skin Cancer
- Sun/Sunlight Damage

</td>
<td>

- Tinea
- Unknown/Normal
- Vascular Tumors
- Vasculitis
- Vitiligo
- Warts

</td>
</tr>
</table>

## 🏗️ Model Architecture

```
Input (224×224×3)
    ↓
Data Augmentation Layer
    ↓
EfficientNetB0 Preprocessing
    ↓
EfficientNetB0 Base Model (frozen → fine-tuned)
    ↓
Global Average Pooling 2D
    ↓
Dropout (0.35)
    ↓
Dense (22 classes, softmax, L2 regularization)
    ↓
Output (22 probabilities)
```

**Training Strategy:**
1. **Stage 1**: Train classifier head with frozen EfficientNetB0 (15 epochs, lr=1e-3)
2. **Stage 2**: Fine-tune top layers of EfficientNetB0 (20 epochs, lr=5e-5)

## 📁 Project Structure

```
SkinDisease/
├── train_model.py              # Complete training pipeline
├── best_skin_model.keras       # Best model checkpoint
├── skin_disease_stage1.keras   # Stage 1 trained model
├── skin_disease_finetuned.keras # Final fine-tuned model
├── class_names.txt             # List of disease classes
├── logs/                       # TensorBoard training logs
├── SkinDisease/               # Dataset directory
│   ├── train/                 # Training images (22 classes)
│   └── test/                  # Validation images (22 classes)
└── SkinWebapp/                # Flask web application
    ├── app.py                 # Flask server
    ├── skin_disease_model.keras # Model for deployment
    ├── class_names.txt        # Class labels
    └── templates/
        └── index.html         # Web interface
```

## 🚀 Installation & Setup

### Prerequisites

- Python 3.8+
- TensorFlow 2.x
- Flask
- NumPy, Pillow, scikit-learn

### Install Dependencies

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install required packages
pip install tensorflow flask numpy pillow scikit-learn
```

## 🎓 Training the Model

### Dataset Preparation

Organize your dataset in the following structure:

```
SkinDisease/
├── train/
│   ├── Acne/
│   ├── Actinic_Keratosis/
│   └── ... (22 classes)
└── test/
    ├── Acne/
    ├── Actinic_Keratosis/
    └── ... (22 classes)
```

### Run Training

```bash
python train_model.py
```

**What happens during training:**
- Loads and preprocesses images (224×224 RGB)
- Computes class weights for imbalanced data
- Trains with data augmentation
- Saves checkpoints and logs to TensorBoard
- Creates three model files: stage1, finetuned, and best checkpoint

### Monitor Training

```bash
tensorboard --logdir=logs
```

## 🌐 Running the Web Application

### Setup

1. Copy the best model to the webapp directory:
```bash
copy best_skin_model.keras SkinWebapp\skin_disease_model.keras
```

2. Ensure `class_names.txt` exists in the webapp directory

### Start the Server

```bash
cd SkinWebapp
python app.py
```

The application will be available at `http://localhost:5000`

### Using the Web Interface

1. Open your browser to `http://localhost:5000`
2. Upload a skin disease image
3. Click "Analyze"
4. View prediction results with:
   - Primary prediction with confidence score
   - Top 5 predictions with probabilities
   - Inference time
   - Image preview

## 🔧 Technical Details

### Preprocessing
- Image resize: 224×224 pixels
- Color mode: RGB
- Preprocessing: EfficientNet-specific normalization

### Data Augmentation
- Horizontal flip
- Random rotation (±20%)
- Random zoom (±20%)
- Random translation (±10%)
- Random contrast adjustment (±20%)

### Training Hyperparameters
- Batch size: 32
- Stage 1 learning rate: 1e-3
- Stage 2 learning rate: 5e-5
- Label smoothing: 0.1 (stage 1), 0.05 (stage 2)
- Dropout rate: 0.35
- L2 regularization: 1e-4
- Optimizer: Adam

### Callbacks
- Early stopping (patience=6, monitor=val_accuracy)
- ReduceLROnPlateau (factor=0.3, patience=3)
- ModelCheckpoint (save best model)
- TensorBoard logging

## 📊 Model Outputs

The trained model provides:
- **Primary Prediction**: Most likely disease class
- **Confidence Score**: Probability of primary prediction (0-100%)
- **Top-5 Predictions**: 5 most likely diagnoses with scores
- **Inference Time**: Processing time in milliseconds

## 🎯 Use Cases

- **Medical Assistance**: Supporting dermatologists in preliminary diagnosis
- **Telemedicine**: Remote skin condition assessment
- **Educational Tool**: Training medical students in dermatology
- **Health Apps**: Integration into mobile health applications
- **Research**: Developing and evaluating ML algorithms for medical imaging

## ⚠️ Disclaimer

**This system is intended for educational and research purposes only.** It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for medical concerns.

## 📝 License

This project is provided as-is for educational and research purposes.

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Adding more disease classes
- Implementing ensemble models
- Improving web interface UX
- Adding confidence threshold filtering
- Deploying to cloud platforms
- Adding explainability features (Grad-CAM, LIME)

## 📧 Contact

For questions or suggestions, please open an issue in the repository.

---

**Note**: Dataset should comply with appropriate medical data usage guidelines and privacy regulations.
