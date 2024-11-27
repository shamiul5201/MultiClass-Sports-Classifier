## MultiClass Sports Classifier
### Overview
The MultiClass Sports Classifier is a machine learning project designed to classify images into multiple sports categories. The code employs a Convolutional Neural Network (CNN) to perform multiclass image classification, demonstrating how to train, validate, and evaluate a deep learning model using TensorFlow/Keras. This project is suitable for image classification tasks where the input consists of images, and the output is one of several predefined sports categories.

![1](https://github.com/user-attachments/assets/3a51d2dd-dfd1-49a9-bef4-d481d89a2289)

This notebook is ideal for:

- **Learning how to preprocess image datasets for deep learning models.**
- **Building custom CNN architectures.**
- **Training and validating multiclass classification models.**
- **Making predictions with trained models.**

## Key Features
- **Custom CNN architecture**: The model is built from scratch to classify sports images into multiple categories.
- **Data Augmentation**: Enhances training by creating variations of the input images.
- **Model Evaluation**: Includes metrics like accuracy and loss.
- **Reusable Workflow**: Can be adapted to other image classification tasks with minimal modifications.

## Code Structure
### 1. Dataset Loading and Preprocessing
This section handles loading the dataset and preprocessing the images. Images are augmented using ImageDataGenerator to improve model generalization.

- **Key Functions**:

1. ImageDataGenerator: Applies transformations like rotation, zoom, and flipping.
2. flow_from_directory: Automatically labels images based on folder structure.

**Example:**
```python
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = train_datagen.flow_from_directory(
    'path_to_train_dataset',
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)
```

### 2. Model Architecture
The CNN model is built using a sequential architecture. It includes convolutional layers for feature extraction, pooling layers for dimensionality reduction, and dense layers for classification.

Architecture Breakdown:

- **Conv2D**: Extracts spatial features using filters.
- **MaxPooling2D**: Reduces the spatial dimensions.
- **Dropout**: Prevents overfitting by randomly deactivating neurons during training.
- **Dense**: Fully connected layers for final classification.

**Example:**
```python
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

```

### 3. Training and Validation
The model is compiled with the Adam optimizer, categorical cross-entropy loss, and accuracy metrics. Training progress is tracked using validation datasets.

**Example:**
```python
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=25
)
```
####  Training Hyperparameters:

* Epochs: `90`

* Learnig Rate: `0.001`

* Batch size: `32`

#### Training History Visualization

<img width="1026" alt="Screenshot 2024-11-08 at 12 18 48 am" src="https://github.com/user-attachments/assets/792b22c2-8aa0-4c1d-bdbd-dc46b63009b8">


### 5. Evaluation and Prediction
The trained model is evaluated on test data, and predictions are made on unseen images.

#### Sample Predictions Visualization

<img width="939" alt="Screenshot 2024-11-08 at 1 10 24 am" src="https://github.com/user-attachments/assets/1fb5196b-9eb4-4683-9454-472b9a5ef9dc">


## Common Pitfalls
- **Class Imbalance**: Address class imbalance using techniques like weighted loss or oversampling.
- **Overfitting**: Use dropout, data augmentation, and early stopping to mitigate overfitting.
- **Input Shape Mismatch**: Ensure all input images are resized to the same dimensions as expected by the model.
- **Learning Rate**: Monitor training; if the loss plateaus, consider adjusting the learning rate.

## Prerequisites and Dependencies
* Python 3.7 or higher
* TensorFlow 2.x
* Matplotlib for visualization
* A GPU is highly recommended for faster training.














