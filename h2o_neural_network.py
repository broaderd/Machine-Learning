import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from keras.callbacks import EarlyStopping
np.set_printoptions(precision = 3)
print("Dillon Broaders 16324857")
print("TensorFlow version:", tf.__version__)

#reads coordinate file and returns list of molecular coordinates
def read_coordinates(file_name):
    coords, c = [], []
    with open(file_name) as f:
        for line in f:
            if line[0] == 'H' or line[0] == 'O':
                l = line.split()
                c.append(float(l[1]))
                c.append(float(l[2]))
                c.append(float(l[3]))
                if len(c) == 9:
                    coords.append(c)
                    c = []
    return coords

#reads the energy file and returns an array of molecular energies (i.e., the labels)
def read_energy(file_name):
    energy = []
    with open(file_name) as f:
        for line in f:
            l = line.strip()
            energy.append(float(l))
    energy_arr = np.array(energy, dtype=float)
    return energy_arr

#calculating the interatmic distance for each molecule (i.e., the features)
def calculate_distance(coordinates):
    distances = np.zeros(shape=(len(coordinates), 3))
    for i, coords in enumerate(coordinates):
        oh1 = np.sqrt((coords[0] - coords[3])**2 + (coords[1] - coords[4])**2 + (coords[2] - coords[5])**2)
        oh2 = np.sqrt((coords[6] - coords[3])**2 + (coords[7] - coords[4])**2 + (coords[8] - coords[5])**2)
        hh = np.sqrt((coords[0] - coords[6])**2 + (coords[1] - coords[7])**2 + (coords[2] - coords[8])**2)
        distances[i] = [oh1, oh2, hh]
    return distances

#model architecture
def model(node1, node2, node3, reg):
    model = keras.models.Sequential([
    keras.layers.Input(shape = (3,)),
    keras.layers.Dense(node1, activation = "relu", kernel_regularizer = keras.regularizers.l2(reg), 
                       bias_regularizer = keras.regularizers.l2(reg)),
    keras.layers.Dense(node2, activation = "relu", kernel_regularizer = keras.regularizers.l2(reg), 
                       bias_regularizer = keras.regularizers.l2(reg)),
    keras.layers.Dense(node3, activation = "relu", kernel_regularizer = keras.regularizers.l2(reg), 
                       bias_regularizer = keras.regularizers.l2(reg)),                 
    keras.layers.Dense(1, kernel_regularizer = keras.regularizers.l2(reg), use_bias=False)])
    return model

#determine learning curve
def learning_curve(model, x_train, y_train, x_test, y_test, training_size, epochs, n_avg):
    train_errors, val_errors = [], []
    print("generating learning curve")
    for m in training_size:
        print("-->training set size in progress:", m)
        train_avg = 0
        val_avg = 0
        for n in range(n_avg):
            model.fit(x_train[:m], y_train[:m], epochs=epochs,  verbose = 0)
            y_train_predict = model.predict(x_train[:m], verbose = 0)
            y_val_predict = model.predict(x_test, verbose = 0)
            train_mse = mean_squared_error(y_train[:m], y_train_predict)
            val_mse = mean_squared_error(y_test, y_val_predict)
            train_avg += train_mse
            val_avg += val_mse
        train_avg /= n_avg
        val_avg /= n_avg
        train_errors.append(train_avg)
        val_errors.append(val_avg)
    return np.sqrt(train_errors), np.sqrt(val_errors)

#plotting predicitions
def plot_predictions(y_test, predictions, ax, file_name):
    ax.scatter(y_test, predictions, color = "blue", alpha = 0.7)
    ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], ls = "--", c = "black")
    ax.set_title(f"H2O energy predictions for {file_name}")
    ax.set_xlabel("True energy values")
    ax.set_ylabel("Predicted energy values")
    ax.grid()

#plotting loss vs epochs
def plot_loss(training_loss, validation_loss, ax, file_name):
    ax.plot(training_loss, label = "Training loss", c = "blue")
    ax.plot(validation_loss, label = "Validation loss", c = "orange")
    ax.set_title(f"Model loss for {file_name}")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss - MSE")
    ax.grid()
    ax.legend()

#plotting learning curve
def plot_learning_curve(training_points, train_errors, val_errors, ax, file_name):
    ax.plot(training_points, train_errors, "-o", label = "Training")
    ax.plot(training_points, val_errors, "-o", label = "Validation")
    ax.set_title(f"Learning curve for {file_name}")
    ax.set_xlabel("Training set size")
    ax.set_ylabel("RMSE")
    ax.legend()
    ax.grid()

if __name__ == "__main__":
    #looping through both training data sets and generating the three plots for each
    data_sets = [("H2O_unrotated.xyz", "H2O_unrotated.ener"), ("H2O_rotated.xyz", "H2O_rotated.ener")]
    fig, axes = plt.subplots(3, len(data_sets), figsize = (15, 15))
    for i, (coord_file, energy_file) in enumerate(data_sets):
        #reading the training/test files
        training_coords = read_coordinates(file_name = coord_file)
        test = read_coordinates(file_name = "H2O_test.xyz")
        #defining the training and test data 
        x_train = calculate_distance(training_coords)
        y_train = read_energy(file_name = energy_file)
        x_test = calculate_distance(test)
        y_test = read_energy(file_name = "H2O_test.ener")
        #scale the features 
        scalar_x = StandardScaler()
        scalar_x.fit(x_train)
        x_train_scaled = scalar_x.transform(x_train)
        x_test_scaled = scalar_x.transform(x_test)
        #scale the labels
        scalar_y = StandardScaler()
        scalar_y.fit(y_train.reshape(-1, 1))
        y_train_scaled = scalar_y.transform(y_train.reshape(-1, 1))
        y_test_scaled = scalar_y.transform(y_test.reshape(-1, 1))
        #compile the model
        epochs = 500
        early_stopping = EarlyStopping(monitor = "val_loss", patience = 10, 
                                    restore_best_weights = True)
        model_ = model(32, 16, 8, 1e-3)
        model_.compile(optimizer = keras.optimizers.Adam(learning_rate = 0.002),
                    loss = "mse",
                    metrics = ["mse"])
        #fit model
        history = model_.fit(x_train_scaled, y_train_scaled, 
                            epochs = epochs, 
                            batch_size = 32, 
                        validation_split = 0.20,
                        callbacks=[early_stopping])
        #calculate loss values
        training_loss = history.history["loss"]
        validation_loss = history.history["val_loss"]
        #get predicitions
        predictions = model_.predict(x_test_scaled)
        y_val = scalar_y.inverse_transform(predictions)
        #generating the learning curve (RMSE as function of training set size)
        training_size = [50, 100, 250, 500, 750, 1000, 1500, 1749]
        train_errors, val_errors = learning_curve(model_, x_train_scaled, y_train_scaled, 
                                    x_test_scaled, y_test_scaled, training_size, epochs = epochs, n_avg = 10)
        #plotting curves
        plot_predictions(y_test, y_val,axes[0, i], energy_file)
        plot_loss(training_loss, validation_loss,axes[1, i], energy_file )
        plot_learning_curve(training_size, train_errors, val_errors, axes[2, i], energy_file)
    plt.tight_layout()
    plt.savefig("subplots.png", dpi = 400)
    plt.show()







