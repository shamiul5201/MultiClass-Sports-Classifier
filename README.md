# <span style="color:blue">MultiClass-Sports-Classifier</span>

![1](https://github.com/user-attachments/assets/3a51d2dd-dfd1-49a9-bef4-d481d89a2289)

### Table of Contents
- [Overview](#overview)
- [Project Motivation](#project-motivation)
- [Imported Libraries and Set Seeds](#imported-libraries-and-set-seeds)
- [Dataset structure](#dataset-structure)
- [Data Preparation](#data-preparation)
- [Data Augmentation and Loading](#data-augmentation-and-loading)
- [Training Configuration](#training-configuration)
- [Callback Setup](#callback-setup)
- [Log Directory Setup](#log-directory-setup)
- [Training History Visualization](#training-history-visualization)
- [Model Architecture: EfficientNetB0-Based Classifier](#model-architecture-efficientnetb0-based-classifier)
- [Model Training Report](#model-training-report)
- [Sample Predictions Visualization](#sample-predictions-visualization)
- [Generating Test Set Predictions and Saving to CSV](#generating-test-set-predictions-and-saving-to-csv)
- [Conclusion](#conclusion)


## <span style="color:red">Overview</span>
This project focuses on building a **deep learning model** to classify images into one of **73 sports categories**. Using a diverse dataset of sports images, the goal is to achieve efficient and accurate classification through a **multi-class image classification model**. This classifier is built with a robust architecture and optimized for handling a wide range of sports, from team sports to individual athletic events. The model’s potential applications include **automated sports categorization for media content, sports analytics, and real-time classification** in various sports-related scenarios.

## <span style="color:red">Project Motivation</span>
The motivation behind this project was to deepen my expertise in **image classification** using **TensorFlow** and **Keras**. With the growing importance of computer vision in fields like sports analytics, media, and real-time image recognition, I wanted to tackle the challenge of building an effective model for multi-class classification. By working on this project, I aimed to enhance my skills in handling complex datasets, designing and tuning neural networks, and implementing best practices in deep learning. This project not only allowed me to improve my technical skills but also opened the potential to apply these models in practical, real-world scenarios within sports and beyond.

### <span style="color:blue">Imported Libraries and Set Seeds</span>
In this step, I imported the essential libraries required for data processing, model building, and visualization:
- **os, random, and numpy:** for data handling and numerical operations.
- **PIL (Python Imaging Library):** to handle images in Python.
- **TensorFlow and Keras:** to construct, train, and manage deep learning models.
- **Matplotlib:** for visualizing data and results.
I also defined a function to set random seeds to ensure **reproducibility**. By fixing random seeds across Python, NumPy, and TensorFlow, I ensured that model training and data processing would yield consistent results upon each run. Additionally, I configured GPU memory growth settings, if available, to prevent memory allocation issues, allowing for stable model training on GPUs.

### <span style="color:blue">Dataset structure</span>
The dataset contains **11,146** images across **73** sports classes. It is split into a training set and a test set:
**train.csv**: Includes image IDs and their corresponding class labels for training the model.
**test.csv**: Contains image IDs for the test set, with no labels, used for generating predictions.
**sports_images/**: Directory containing all the images, named by their ID.

**Training**: Use train.csv for training, and split it into train and validation sets (usually 80:20 ratio).
**Testing**: For the test set, use test.csv to generate predictions after training. Submit a .csv file with image ID and predicted class.

### <span style="color:blue">Data Preparation</span>
In this section, I prepared the dataset for training by splitting it into training and validation sets, as no separate validation set was provided. To achieve this, I divided the training data in an 80:20 ratio, creating an 80% training and 20% validation split.
- **1. Loaded the Dataset**: I loaded the dataset from a CSV file that contained image IDs and corresponding class labels.
- **2. Defined Directory Structure**: I organized the dataset into a structured folder format with separate directories for training and validation data. I created subfolders for each class within both the training and validation directories.
- **3. Data Splitting**: Using train_test_split from scikit-learn, I divided the dataset into training and validation subsets, ensuring that the class distribution was maintained. This step allowed me to replicate a validation set to assess model performance on unseen data during training.
- **4. Organized and Moved Images**: I wrote a function to move images into their respective class folders within the training and validation directories. This helped ensure that each image was available in a structured, class-based format for model input.
- **5. Validation**: To verify the structure, I printed the directory layout and the number of samples in each split. Additionally, I counted and displayed the total number of images in the training and validation folders to confirm the 80:20 split was accurately implemented.
- **6. Sample Image Inspection**: I also randomly selected a sample image from one of the classes to check its dimensions, number of channels, and visual appearance. This step was useful in verifying image consistency and display.
<img width="983" alt="Screenshot 2024-11-08 at 12 00 58 am" src="https://github.com/user-attachments/assets/95398035-1d9a-4344-b062-43984b70d1c1">

### <span style="color:blue">Data Augmentation and Loading</span>
To enhance the model’s robustness and help it generalize better, I applied a series of data augmentation techniques and then defined a data loading pipeline. Here’s a breakdown of the process:
- **Data Augmentation**:
I used TensorFlow’s Keras layers to create a custom data **augmentation** pipeline that included:
  - **RandomRotation**: Rotated images by up to 15% to introduce angle variance.
  - **RandomFlip**: Flipped images horizontally to increase data diversity.
  - **RandomZoom**: Applied random zoom transformations, enhancing scale invariance.
  - **RandomContrast and RandomBrightness**: Adjusted contrast and brightness randomly to simulate varying lighting conditions.
These augmentations helped prevent overfitting by diversifying the training data with realistic variations. The augmentation pipeline was implemented as a **tf.keras.Sequential** model to apply these transformations sequentially.
- **Data Loading and Preprocessing**:
  wrote a function to load and preprocess the dataset using TensorFlow's **image_dataset_from_directory** utility. Key steps included:
  - **Dataset Splits**: Loaded images from the train and valid directories, applying categorical labels for multi-class classification.
  - **Batching and Resizing**: Resized images to a fixed target size **(224x224)** and batched them to manage memory usage effectively.
  - **Data Augmentation Integration**: When data augmentation was enabled, I applied the augmentation pipeline only to the training dataset, 
    leaving the validation set unaltered to maintain evaluation consistency.
  - **Prefetching**: Used **tf.data.AUTOTUNE** to prefetch batches, reducing I/O latency and speeding up model training.
Together, these steps established an efficient and flexible data pipeline, ensuring that the training data was varied and that the model could be trained more effectively with minimal bottlenecks.

### <span style="color:blue">Training Configuration</span>
To configure training parameters, I created a **TrainingConfig** dataclass, which defined essential hyperparameters such as **BATCH_SIZE**, **EPOCHS**, and **LEARNING_RATE**. Setting these as class variables enabled easy adjustments and tuning for different experiments.
For tracking progress and saving models, I set up dedicated directories for TensorBoard logs and checkpoints under Logs_Checkpoints, with subfolders for each version. These directories stored training logs and model states, enabling detailed monitoring and version control of model performance. I initialized the logging and checkpoint folders as **version_0** to establish a consistent versioning system.

### <span style="color:blue">Callback Setup</span>
To manage training checkpoints and monitor metrics, I defined a get_callbacks function. This function used two key callbacks: **TensorBoard and ModelCheckpoint**.
**The TensorBoard callback** logged the model’s performance at each epoch, recording metrics and creating visualizations that helped in understanding the training dynamics. I also set it to update images and histograms periodically, making it easier to spot overfitting or underfitting early on.
**The ModelCheckpoint callback** saved model weights based on performance, updating the file path depending on whether save_weights_only was True or False. With the save_best_only option, it kept only the best-performing model, ensuring I had the top model version from each run. These callbacks streamlined the tracking and model-saving processes, ensuring that the most accurate version was always available.

### <span style="color:blue">Log Directory Setup</span>
For managing logging and checkpoints, I defined the **setup_log_directory** function, which dynamically created directories for TensorBoard logs and model checkpoints. This setup ensured that each experiment was saved in its unique folder, helping with version tracking and organization. I implemented a **get_number** function to extract version numbers from existing folders and then identified the latest version in the log directory. If a log directory already existed, I **incremented** the version number for a new folder name. Otherwise, it defaulted to version_0. This automatic versioning kept each experiment’s data distinct and easy to revisit. After determining the version, I updated the paths in **training_config** for both logs and checkpoints, creating the necessary directories if they didn’t already exist. This setup made the logging process smooth and consistent across experiments.

### <span style="color:blue">Training History Visualization</span>
To analyze model performance visually, I implemented the **plot_history** function, which plotted both the loss and accuracy metrics for training and validation. This function accepted loss and **metric data** for both training and validation sets, allowing me to track improvements and spot issues like **overfitting**.
In the loss plot, training and validation loss curves were displayed together, making it easy to see if the model’s validation performance was tracking well alongside training. Similarly, the accuracy plot compared training and validation accuracy across epochs, helping to ensure the model was learning effectively.
I included customization options like **color**, **legend location**, and **figure size** to allow flexibility in how results were displayed. Each plot used grids and clear labels for readability, creating a straightforward way to monitor model progress over time.

<img width="1026" alt="Screenshot 2024-11-08 at 12 18 48 am" src="https://github.com/user-attachments/assets/792b22c2-8aa0-4c1d-bdbd-dc46b63009b8">

## <span style="color:blue">Model Architecture: EfficientNetB0-Based Classifier</span>
This model leverages **EfficientNetB0** as a pre-trained backbone, which is a highly efficient **convolutional neural network** optimized for a balance between **accuracy** and **computational cost**. Below is a detailed description of each part of the model architecture:
- **Base Model: EfficientNetB0 (Pre-trained on ImageNet)**
I loaded the EfficientNetB0 model with pre-trained weights from **ImageNet**, which allowed the model to benefit from powerful, pre-existing feature representations. Setting **include_top=False** removed the default classification head, enabling the model to be customized for a different number of output classes.
- **Freezing the Base Model**: By setting **base_model.trainable = False**, I ensured that the weights of the EfficientNetB0 layers were not updated during training. This approach helped retain the general visual features learned from ImageNet, preventing overfitting on smaller datasets or cases where computation is limited.
- **GlobalAveragePooling2D Layer**
After the EfficientNetB0 backbone, I added a **GlobalAveragePooling2D** layer to condense the spatial dimensions of the feature maps into a single vector per feature channel. This pooling layer provided a compact representation of the learned features without adding many parameters, making it computationally efficient.
- **Dense Layer (1024 units, ReLU Activation)**
To give the model more learning capacity, I included a Dense layer with 1024 units and a **ReLU activation function**. This fully connected layer helped learn complex patterns and refined the feature representations from EfficientNetB0, making the model better at classifying specific categories.
- **Dropout Layer (0.5 Rate)**
The Dropout layer, with a rate of 0.5, added a form of **regularization** to reduce overfitting. During each training step, it randomly set 50% of the units in the previous dense layer to zero, forcing the model to rely on diverse patterns rather than memorizing specific ones. This approach improved the model’s generalization performance on unseen data.
- **Output Layer (Dense, Softmax Activation)**
The final layer was a Dense layer with **num_classes** units, each representing a class in the classification task. The softmax activation function converted the outputs into class probabilities, allowing the model to predict the likelihood of each class for a given input.
### Summary of the Model Design
This EfficientNetB0-based model combined pre-trained features with a customized, simple classification head to create a lightweight yet powerful classifier. The architecture benefited from the rich feature extraction of EfficientNetB0 while maintaining flexibility in the final layers to learn specific class distinctions. With regularization via dropout and efficient pooling, this model was optimized for performance and generalization on new data.
<img width="1015" alt="Screenshot 2024-11-08 at 12 30 20 am" src="https://github.com/user-attachments/assets/d5cc78af-845e-4aa0-acbb-5e4b6fbe7dae">

### <span style="color:blue">Model Training Report</span>
The model was trained over 90 epochs with the following key metrics observed across the training process:
- **Epoch 1**:
  - **Training Accuracy**: 0.2623
  - **Training Loss**: 3.0651
  - **Validation Accuracy**: 0.7942
  - **Validation Loss**: 0.7308
The model rapidly improved within the initial epochs, reaching over 85% validation accuracy by Epoch 5 with a significant reduction in validation loss. After further optimization, training yielded the following notable results:
- **Epoch 10**:
  - **Accuracy**: 0.7646
  - **Loss**: 0.8022
  - **Validation Accuracy**: 0.8894
  - **Validation Loss**: 0.3712
Further increments in accuracy and validation metrics were observed up until around Epoch 25. Past this point, fluctuations in validation accuracy and loss were noted, indicating potential overfitting or need for regularization adjustments.
- **Epoch 50**:
  - **Accuracy**: 0.8479
  - **Loss**: 0.5385
  - **Validation Accuracy**: 0.9058
  - **Validation Loss**: 0.4272
By the end of training:
- **Final Epoch (90)**:
  - **Accuracy**: Achieved over 85% consistently
  - **Validation Accuracy**: Reached stability around 90%
  - **Loss values demonstrated stability with slight fluctuations due to batch variability**
This training report suggests the model effectively learned the task with room for further refinement in regularization and hyperparameter tuning to enhance generalization on the validation set.

### <span style="color:blue">Sample Predictions Visualization</span>
The **get_sample_predictions** function provides a visual overview of the model's performance on the validation dataset by displaying predicted and true class labels for a selected number of samples. It performs the following steps:
- **Prediction Generation**: For each image in the validation dataset, the function uses the model to predict the class and corresponding probability. The highest-probability prediction is identified as the model's choice for the image's class.
- **Image and Label Display**: The function arranges a grid of sample images, displaying both the model's predicted class **(P)** and the true class label **(T)** in the title above each image. Prediction probability scores are also shown, allowing for an assessment of confidence in each prediction.
- **Configuration and Output**: You can specify the total number of images displayed **(default is 15)**.
This visualization aids in quickly identifying patterns in **correct** and **incorrect** predictions, providing insights into class-specific model strengths and weaknesses.
<img width="939" alt="Screenshot 2024-11-08 at 1 10 24 am" src="https://github.com/user-attachments/assets/1fb5196b-9eb4-4683-9454-472b9a5ef9dc">

### <span style="color:blue">Generating Test Set Predictions and Saving to CSV</span>
This script leverages a trained model to predict labels for a test dataset of sports images and outputs the results in a CSV file format. It performs the following steps:
- **Import Necessary Libraries**: Imports essential libraries such as pandas, numpy, and tensorflow to handle data processing, model loading, and image handling.
- **Configuration Setup**: Uses DatasetConfig and TrainingConfig (assumed to be defined elsewhere) to set paths and parameters required for data and model configurations.
- **Define Class Labels**: A list of class labels is defined, representing the different sports categories that the model can predict.
- **Load Test Data**: Reads the test.csv file containing image IDs and other relevant metadata needed for making predictions.
- **Load Trained Model**: The trained model is loaded from a specified checkpoint path, allowing for direct use in the prediction phase.
- **Image Prediction Loop**: For each image ID in test.csv, the script:
  - **Loads and preprocesses the image, resizing it to the target input size expected by the model**.
  - **Makes a prediction using the trained model, then maps the prediction to the corresponding class label**.
  - **Adds the prediction result to a list for future storage**.
- **Save Predictions to CSV**: After all predictions are generated, the results are saved in a new CSV file named submission.csv, containing columns for ID and CLASS, which store the image ID and predicted class label, respectively.
This workflow is designed for efficient batch predictions and provides a straightforward way to evaluate model performance on an unseen test set by generating a submission file.

### <span style="color:blue">Conclusion</span>
This project successfully implemented an image classification model capable of identifying a wide range of sports categories from visual data. Through careful dataset preparation, model selection, and tuning, the final model achieved strong predictive performance, demonstrating its potential applicability in real-world sports image classification tasks.
While the results are promising, some challenges were encountered, such as **[briefly mention any challenges, like class imbalance or computational limitations if relevant]**. Addressing these in future work could enhance model accuracy and robustness further.
Future improvements could focus on:
- **Model Fine-tuning**: Experimenting with alternative architectures or incorporating more advanced fine-tuning techniques to improve classification accuracy.
- **Augmentation and Data Diversity**: Expanding the dataset with more diverse and balanced samples to mitigate any biases and further enhance generalization.
- **Real-World Application**: Integrating the model into applications where real-time sports classification is required, like streaming platforms or event highlight detectors.
Overall, this project illustrates a complete pipeline for building, training, and deploying a sports image classifier, providing a solid foundation for further exploration in the field of visual recognition within sports.


















