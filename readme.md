# Image Geolocation Prediction Using k-NN and CLIP Embeddings

## Overview
This project predicts the geographic coordinates (latitude and longitude) of images using the k-Nearest Neighbors (k-NN) algorithm. The dataset comprises geo-tagged images from Flickr, and we leverage OpenAI's CLIP embeddings for image feature analysis.

## Key Features
- Predicts image locations using k-NN.
- Employs PCA for dimensionality reduction.
- Optimizes k-NN with grid search for the best k-value.
- Evaluates performance using Mean Displacement Error (MDE).

## Requirements
The following Python libraries are required:
- matplotlib
- numpy
- scikit-learn

## Installation
1. Clone the repository
2. Navigate to the project directory
3. Install dependencies

## Usage
1. Place the dataset file `im2spain_data.npz` in the project directory.
2. Run the main script:


## Output
- Visualizations of image locations and features.
- Mean Displacement Error for different k-values.
- Comparison of k-NN with Linear Regression.

## Notes
Ensure that the required Python libraries are installed before running the code. For any issues, please contact the project maintainer.
