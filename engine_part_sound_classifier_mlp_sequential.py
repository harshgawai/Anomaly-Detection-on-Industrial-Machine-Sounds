import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import matplotlib.pyplot as plt
dataset_path = "data.json"

def load_data(dataset_path):
    with open(dataset_path, "r") as fp:
        data = json.load(fp)

        inputs = np.array(data["mfcc"])
        targets = np.array(data["labels"])

        return inputs, targets

def plot_history(history):
    fig, axes = plt.subplots(2)

    axes[0].plot(history.history["accuracy"], label="train accuracy")
    axes[0].plot(history.history["val_accuracy"], label="test accuracy")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend(loc="lower right")
    axes[0].set_title("Accuracy eval")

    axes[1].plot(history.history["loss"], label="train error")
    axes[1].plot(history.history["val_loss"], label="test error")
    axes[1].set_ylabel("Error")
    axes[1].set_xlabel("Epoch")
    axes[1].legend(loc="upper right")
    axes[1].set_title("Error eval")

    plt.show()

if __name__ == "__main__":
    inputs, targets = load_data(dataset_path)
    print(inputs.shape)
    X_train, X_test, y_train,  y_test = train_test_split(inputs, targets, test_size=0.3)

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(inputs.shape[1], inputs.shape[2])), #input layer

        keras.layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)), #hidden layer
        keras.layers.Dropout(0.3),

        keras.layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)), #hidden layer
        keras.layers.Dropout(0.3),

        keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)), #hidden layer
        keras.layers.Dropout(0.3),

        keras.layers.Dense(3,activation="softmax") #Output layer
    ])

    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    history = model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=50, batch_size=32)

    plot_history(history)