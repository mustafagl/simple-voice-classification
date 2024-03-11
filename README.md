# Simple Voice Classification

## Overview
This notebook demonstrates a simple approach to classifying voice commands in the Turkish language using a Convolutional Neural Network (CNN) model. The dataset used is the Turkish Speech Command Dataset, which contains spoken commands for various actions.

## Data Preprocessing
- The audio files are read and converted into numerical representations using the `scipy.io.wavfile` module.
- The audio data is normalized to have values between -1 and 1.
- The dataset is split into training and testing sets using a 80-20 split.

## Model Architecture
- The model is a CNN with two convolutional layers followed by max-pooling and dropout layers to reduce overfitting.
- The first convolutional layer has 128 filters with a kernel size of 64 and a stride of 4.
- The second convolutional layer has 256 filters with a kernel size of 32 and a stride of 4.
- Global average pooling is used to reduce the dimensionality before the final dense layer.
- The output layer has 15 units corresponding to the 15 different commands in the dataset.

## Training
- The model is trained for 15 epochs with a batch size of 64.
- The Adam optimizer is used with the default learning rate.
- The loss function is Sparse Categorical Crossentropy as the labels are integers.
- The accuracy metric is used to evaluate the model's performance.

## Results
- The model achieves an accuracy of approximately 91.15% on the training set and 88.78% on the test set.

## Dependencies
- TensorFlow
- Keras
- NumPy
- SciPy
- scikit-learn

## Usage
- Ensure all dependencies are installed.
- Download the Turkish Speech Command Dataset and update the `basepath` variable in the notebook.
- Run the notebook cells sequentially to preprocess the data, build the model, and train it.

## Acknowledgements
- Dataset: [Turkish Speech Command Dataset](https://www.kaggle.com/datasets/yavuzsencan/turkish-speech-command-dataset)

## Author
- [Mustafa Gull](https://www.kaggle.com/mustafagull) (Kaggle Profile)
