# Traffic Sign Classification Using Deep Learning

## Project Overview
This project focuses on building a convolutional neural network (CNN) to classify traffic signs from the [German Traffic Sign Recognition Benchmark (GTSRB)](http://benchmark.ini.rub.de/). The model is trained on thousands of images of traffic signs, which are then used to predict the type of sign from unseen images.

## Features
- **Data Preprocessing**: Includes converting images to grayscale, normalization, and augmentation.
- **CNN Model**: A deep learning model using TensorFlow and Keras.
- **Evaluation**: Accuracy measurement and a confusion matrix to visualize the model performance.
- **Prediction**: Predicting traffic signs from new test images to evaluate the effectiveness of the trained model.

## Prerequisites
Before you begin, ensure you have met the following requirements:
- Python 3.8 or higher
- TensorFlow 2.x
- NumPy, Pandas, Matplotlib, Seaborn
- Sklearn (for shuffling and confusion matrix)

You can install the required packages using pip:

```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn
```

## Dataset
The dataset used is the German Traffic Sign Recognition Benchmark (GTSRB), which you can download [here](http://benchmark.ini.rub.de/Dataset_GTSRB_Final_Training_Images.zip). Unzip the dataset into a folder named `traffic-signs-data` in the root directory of the project.

## Setup and Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/traffic-sign-classification.git
   ```
2. Navigate to the project directory:
   ```bash
   cd traffic-sign-classification
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
To run the project, execute the following command in the project root directory:
```bash
python traffic_sign_classifier.py
```

## Model Architecture
The model consists of the following layers:
- Convolutional Layer 1: 6 filters, 5x5, activation 'relu'
- Dropout Layer: dropout rate of 0.2
- Convolutional Layer 2: 16 filters, 5x5, activation 'relu'
- Average Pooling Layer: pool size 2x2
- Flatten Layer: to flatten the output for the dense layer
- Dense Layer 1: 120 units, activation 'relu'
- Dense Layer 2: 84 units, activation 'relu'
- Output Layer: 43 units, activation 'softmax' (one unit for each traffic sign type)

## Results
- **Training Accuracy**: XX%
- **Validation Accuracy**: XX%
- **Test Accuracy**: XX%

## Contributing to the Project
If you want to contribute to the development of this project, please follow these steps:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature_branch_name`).
3. Make changes and commit your changes (`git commit -am 'Add some feature'`).
4. Push to the branch (`git push origin feature_branch_name`).
5. Create a new Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE.md file for details.

## Acknowledgments
- [German Traffic Sign Recognition Benchmark (GTSRB)](http://benchmark.ini.rub.de/)
- Inspiration from various deep learning projects and community forums.
