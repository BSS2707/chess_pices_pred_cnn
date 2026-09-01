# ♟️ Chess Piece Recognition System

An amazing AI-powered **Streamlit web application** that uses OpenCV and CNN (Convolutional Neural Network) to recognize chess pieces in real-time!

## Features

✨ **Real-time Camera Detection**
- Open your webcam directly in the app
- Continuous chess piece recognition
- Instant accuracy percentage display
- Confidence visualization with progress bar
- Perfect for testing your model live

📁 **Image Upload Analysis**
- Upload chess piece images via web interface
- Instant CNN classification
- Detailed confidence metrics and statistics
- Supports PNG, JPG, JPEG formats

🎯 **6 Chess Pieces Supported**
- ♔ King
- ♕ Queen
- ♖ Rook
- ♗ Bishop
- ♘ Knight
- ♙ Pawn

🌐 **Web-Based Interface**
- Beautiful, modern UI with Streamlit
- Responsive design
- Sidebar navigation
- Real-time results
- No installation of extra GUI libraries needed

## Installation

### Prerequisites
- Python 3.8 or higher
- Webcam (optional, for camera features)
- Modern web browser (Chrome, Firefox, Safari, Edge)

### Setup

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Prepare your dataset:**
   - Ensure your `dataset/` folder has subdirectories for each chess piece:
   ```
   dataset/
   ├── Bishop/
   ├── King/
   ├── Knight/
   ├── Pawn/
   ├── Queen/
   └── Rook/
   ```
   - Add training images to each directory (100+ per class recommended)

## Usage

### First Run (Train the Model)
```bash
python model.py
```
This will:
- Load images from the `dataset/` folder
- Train a CNN model on your chess piece images
- Save the trained model as `chess_piece_model.h5`
- Display training metrics and test accuracy

### Run the Web Application
```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

**Choose a mode from the sidebar:**
1. **📷 Camera Detection** - Real-time detection from your webcam
2. **📁 Image Upload** - Analyze saved image files
3. **ℹ️ About** - View system information and tips

## Model Architecture

The CNN model includes:
- **3 Convolutional Blocks** with BatchNormalization
- **Max Pooling** layers for dimensionality reduction
- **Dropout** layers (25-50%) for regularization
- **Global Average Pooling** for feature aggregation
- **Dense layers** with Dropout for classification
- **Softmax activation** for multi-class probability

### Model Specifications
- **Input Size**: 224×224×3 RGB
- **Number of Classes**: 6 (one per chess piece)
- **Training Epochs**: 25
- **Batch Size**: 32
- **Optimizer**: Adam (learning_rate=0.001)
- **Loss Function**: Sparse Categorical Crossentropy

## Output Format

Example predictions shown in app:

**Camera Mode:**
```
✅ Detection Successful!
♟️ Detected: King
Accuracy: 95.3%
[████████░░░░░░░░░░] Progress bar
```

**Image Upload Mode:**
```
✅ Detection Successful!
♟️ King
Accuracy: 95.3%
Piece: KING | Confidence Score: 95.3% | File: chess_king.jpg | Reliability: ✅ High
```

## File Structure

```
chess_cnn_Ai/
├── model.py              # CNN model definition and training
├── app.py                # Streamlit web application
├── requirements.txt      # Python dependencies
├── README.md             # This file
├── chess_piece_model.h5  # Trained model (generated after training)
└── dataset/              # Training images
    ├── Bishop/
    ├── King/
    ├── Knight/
    ├── Pawn/
    ├── Queen/
    └── Rook/
```

## Tips for Best Results

### 1. **Training Data Quality**
   - Use clear, well-lit images of chess pieces
   - Varied angles, distances, and backgrounds improve accuracy
   - 100-200 images per class recommended
   - Mix different lighting conditions

### 2. **Camera Usage**
   - Ensure good lighting conditions
   - Keep chess piece centered and clearly visible
   - Avoid shadows and bright glare
   - Use a steady hand or tripod

### 3. **Interpreting Results**
   - Confidence > 70% indicates highly reliable detection
   - Confidence 50-70% suggests possible misclassification
   - Confidence < 50% likely indicates poor image quality
   - Confidence bar visualization helps at a glance

### 4. **Improving Model Accuracy**
   - Train with more diverse images
   - Increase EPOCHS in model.py
   - Use data augmentation (rotation, brightness, etc.)
   - Adjust BATCH_SIZE for your system

## Troubleshooting

### Camera not accessible
```
Error: "Camera not available"
```
- Check if your webcam is connected
- Verify no other app is using the camera
- Try refreshing the browser
- Check browser permissions for camera access

### Low prediction accuracy
- Increase dataset size and diversity
- Improve lighting conditions during training
- Increase EPOCHS in model.py (try 50-100)
- Ensure training images are clear and centered

### Model file not found
```
Error: "Model not found at chess_piece_model.h5"
```
- Run `python model.py` first to train the model
- Check that file exists in the project root
- Ensure training completed successfully

### Out of memory errors
- Reduce BATCH_SIZE in model.py
- Use smaller images (reduce IMG_SIZE)
- Close other applications
- Use a GPU-enabled TensorFlow if available

## Advanced Usage

### Retrain the model
```bash
# Delete the old model
rm chess_piece_model.h5

# Add new training images to dataset folders
# Then retrain
python model.py
```

### Adjust model hyperparameters
Edit variables in `model.py`:
```python
IMG_SIZE = 224        # Image dimensions
BATCH_SIZE = 32       # Training batch size
EPOCHS = 25           # Number of training epochs
```

### Deploy Streamlit app online
```bash
# Using Streamlit Cloud
streamlit deploy app.py
```

## Performance Metrics

Typical performance on well-trained models:
- **Average Accuracy**: 85-95%
- **Training Time** (100 images/class): 2-5 minutes
- **Inference Time**: <100ms per image
- **Model Size**: ~60-80 MB

## Future Enhancements

- [ ] Board recognition and piece position mapping
- [ ] Multi-piece detection in single frame
- [ ] Model accuracy metrics dashboard
- [ ] Export detected positions to chess notation
- [ ] Video recording with annotations
- [ ] Model ensemble for higher accuracy
- [ ] Transfer learning with pre-trained models
- [ ] Real-time performance profiling

## Technologies Used

- **Streamlit** - Web framework for data apps
- **TensorFlow/Keras** - Deep learning framework
- **OpenCV** - Computer vision library
- **NumPy** - Numerical computing
- **Scikit-learn** - Machine learning utilities
- **Pillow** - Image processing

## License

This project is open-source and free to use and modify for personal and commercial projects.

---

**Built with ♟️ and AI** 🤖

**Enjoy recognizing chess pieces!** ✨
