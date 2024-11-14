# <span style="color:blue">MultiClass-Sports-Classifier</span>
This project is all about creating a deep learning model that can recognize and classify images into 73 different sports categories. Using a wide variety of sports images, the aim is to build a model that’s both accurate and efficient, capable of distinguishing between everything from team sports to solo athletic events.

### Dataset overview
The dataset consists of 11,146 images, covering 73 different sports categories.
![1](https://github.com/user-attachments/assets/3a51d2dd-dfd1-49a9-bef4-d481d89a2289)

Out of the 11,146 images, 8,917 are used for training the model, while the remaining 2,229 images are part of the test set, which will be used for generating predictions after the model has been trained.
For the training process, the data is further split into an 80-20 ratio, with 80% of the images used for training and 20% reserved for validation.

### <span style="color:blue">Model Architecture: EfficientNetB0-Based Classifier</span>
The model uses EfficientNetB0 as its pre-trained backbone—a highly efficient convolutional neural network that strikes a good balance between accuracy and computational efficiency.

<img width="1015" alt="Screenshot 2024-11-08 at 12 30 20 am" src="https://github.com/user-attachments/assets/d5cc78af-845e-4aa0-acbb-5e4b6fbe7dae">

### Data Augmentation
The following data augmentation were used to address overfitting

```python
def data_augmentation_preprocess(seed=None):
    if seed is not None:
        tf.random.set_seed(seed)

    rand_rotation = RandomRotation(0.15, fill_mode='nearest', seed=seed)
    rand_flip_hor = RandomFlip("horizontal", seed=seed)
    rand_zoom = RandomZoom(height_factor=(-.5, .5), width_factor=(-.5, .5), fill_mode='nearest', seed=seed)
    rand_contrast = RandomContrast(factor=0.2, seed=seed)
    rand_brightness = RandomBrightness(factor=0.2, seed=seed)

    data_augmentation_pipeline = tf.keras.Sequential([
        rand_rotation,
        rand_flip_hor,
        rand_zoom,
        rand_contrast,
        rand_brightness,
    ])

    return data_augmentation_pipeline
```


###  Training Hyperparameters:

* Epochs: `90`

* Learnig Rate: `0.001`

* Batch size: `32`



### <span style="color:blue">Training History Visualization</span>

<img width="1026" alt="Screenshot 2024-11-08 at 12 18 48 am" src="https://github.com/user-attachments/assets/792b22c2-8aa0-4c1d-bdbd-dc46b63009b8">


### <span style="color:blue">Sample Predictions Visualization</span>

<img width="939" alt="Screenshot 2024-11-08 at 1 10 24 am" src="https://github.com/user-attachments/assets/1fb5196b-9eb4-4683-9454-472b9a5ef9dc">


















