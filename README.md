# Automated Ocular Disease Detection using Deep Learning
This repository contains a collection of Jupyter Notebooks that demonstrate the development and comparative analysis of deep learning models for automated ocular disease detection. The notebooks cover data preprocessing, model training, evaluation, and visualization of results.

## Overview
The project leverages the Ocular Disease Intelligent Recognition (ODIR) dataset, which consists of retinal fundus images annotated for eight ocular conditions. In our analysis, we compare the performance of several state-of-the-art convolutional neural network architectures (such as ResNet50, VGG16, InceptionV3, and DenseNet121) using a unified preprocessing pipeline. Our approach includes image standardization, data augmentation, normalization, and GPU-based mixed precision training to optimize both diagnostic performance and computational efficiency.

## Dataset
You can download the dataset directly from Kaggle:

[ODIR5K Dataset on Kaggle](https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k).

Please ensure you follow the Kaggle instructions to download and extract the dataset, then update the file paths in the notebooks accordingly.

## Repository Contents
### Data_Preprocessing.ipynb
Contains the code for preprocessing the retinal images, including resizing, normalization, and data augmentation.

### Model_Training_ResNet50.ipynb
Demonstrates training and evaluation of the ResNet50 model.

### Model_Training_VGG16.ipynb
Demonstrates training and evaluation of the VGG16 model.

### Model_Training_InceptionV3.ipynb
Demonstrates training and evaluation of the InceptionV3 model.

### Model_Training_DenseNet121.ipynb
Demonstrates training and evaluation of the DenseNet121 model.

## Environment Setup
To run these notebooks, create a virtual environment and install the required packages. For example:

```
conda create -n ocular_env python=3.8
conda activate ocular_env
pip install tensorflow pandas numpy matplotlib scikit-learn tqdm pydot graphviz
```
If you are using GPU acceleration, ensure that you have the compatible NVIDIA drivers, CUDA, and cuDNN installed.

## How to Use
Download the ODIR5K Dataset:
Download the dataset from Kaggle using the provided link and extract it to a folder on your system.

## Update File Paths:
In the notebooks, update the file paths to point to the downloaded dataset location (Code given in ResNet and VGG files).

## Run Notebooks:
Execute the notebooks sequentially:

## Start with data preprocessing.
Then train and evaluate each model.
Finally, run the visualization notebook to analyze the results.
## Contributing
Contributions and suggestions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.
