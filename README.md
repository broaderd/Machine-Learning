# Molecular Energy Prediction with Neural Networks

This repository contains Python code for building, training, and evaluating neural network models to predict the energy levels of water molecules based on their molecular coordinates.

## Requirements

- Python 3.11.5
- NumPy 1.26.4
- Matplotlib 3.8.0
- TensorFlow 2.12.0
- Keras 2.12.0
- scikit-learn 1.4.1.post1

## File Descriptions

- **main.py**: The main script containing the implementation of the neural network model, data reading functions, model training, evaluation, and plotting functions.
- **H2O_unrotated.xyz**: Input file containing molecular coordinates for unrotated water molecules.
- **H2O_unrotated.ener**: Input file containing energy values corresponding to unrotated water molecules.
- **H2O_rotated.xyz**: Input file containing molecular coordinates for rotated water molecules.
- **H2O_rotated.ener**: Input file containing energy values corresponding to rotated water molecules.
- **H2O_test.xyz**: Test dataset containing molecular coordinates for evaluation.
- **H2O_test.ener**: Test dataset containing energy values for evaluation.
- **subplots.png**: Output file containing subplots of predictions, loss, and learning curves.

## Usage

1. Ensure all required dependencies are installed.
2. Place the input data files (`H2O_unrotated.xyz`, `H2O_unrotated.ener`, `H2O_rotated.xyz`, `H2O_rotated.ener`, `H2O_test.xyz`, `H2O_test.ener`) in the same directory as `main.py`.
3. Run `main.py`. The script will train the neural network model, evaluate its performance, and generate plots.
4. The output plots will be saved as `subplots.png`.

## Methodology

1. **Data Preprocessing**: The script reads molecular coordinates and energy values from input files. It then calculates interatomic distances as features and scales both features and labels using `StandardScaler`.

2. **Model Building**: A neural network model is constructed using Keras, consisting of multiple dense layers with ReLU activation and L2 regularization.

3. **Model Training**: The model is trained using Adam optimizer and Mean Squared Error loss. Early stopping is implemented based on validation loss.

4. **Model Evaluation**: The trained model is evaluated on test data, and predictions are compared with true energy values. Loss over epochs and learning curves are plotted for analysis.

## Authors

- Dillon Broaders
