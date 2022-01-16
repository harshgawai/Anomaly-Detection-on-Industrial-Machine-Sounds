import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import matplotlib.pyplot as plt
dataset_path = "data.json"

def load_data(dataset_path):
    with open(dataset_path, "r") as fp:
        data = json.load(fp)

    X = np.array(data["mfcc"])
    y = np.array(data["labels"])

    return X, y
def prepare_dataset(test_size, validation_size):

    X, y = load_data(dataset_path)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)

    return X_train, X_validation, X_test, y_train, y_validation, y_test

def plot_hist(hist):
    fig, axes = plt.subplots(2)

    axes[0].plot(hist.history["accuracy"], label="train accuracy")
    axes[0].plot(hist.history["val_accuracy"], label="test accuracy")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend(loc="lower right")
    axes[0].set_title("Accuracy eval")

    axes[1].plot(hist.history["loss"], label="train error")
    axes[1].plot(hist.history["val_loss"], label="test error")
    axes[1].set_ylabel("Error")
    axes[1].set_xlabel("Epoch")
    axes[1].legend(loc="upper right")
    axes[1].set_title("Error eval")

    plt.show()

def build_model(input_shape):

    model = keras.Sequential() #create model

    model.add(keras.layers.LSTM(64, input_shape=input_shape, return_sequences=True)) #for sequence to sequence
    model.add(keras.layers.LSTM(64)) #ssequence to vector

    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.Dense(3, activation="softmax")) #Output layer

    return model

def predict(model, X, y):
    X = X[np.newaxis, ...]
    prediction = model.predict(X)
    print("Prediction", prediction)
    with open(dataset_path, "r") as fp:
        data = json.load(fp)


    predicted_index = np.argmax(prediction, axis=1) #Output = 1d array
    print("Predicted index",predicted_index)
    pred = predicted_index[0]

    print("Expected class: {}, Predicted class: {}".format(data["mappings"][y], data["mappings"][pred]))



if __name__ == "__main__":
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_dataset(0.25,0.2)

    input_shape = (X_train.shape[1], X_train.shape[2]) #Rnn-lstm takes 2 dimensional input
    model = build_model(input_shape) #build the network

    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"]) #compiled it

    model.summary()

    hist = model.fit(X_train, y_train, validation_data=(X_validation, y_validation), batch_size=32, epochs=2)  #train the model
    plot_hist(hist)

    test_error, test_accuracy = model.evaluate(X_test, y_test, verbose = 1) #evaluate the cnn

    print("Accuracy on test set is: {}".format(test_accuracy))

    X = X_test[110]
    y = y_test[110]

    predict(model, X, y)