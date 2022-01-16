import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
dataset_path = "data.json"
#dataset_ab = "data3.json"
def load_data(dataset_path):
    with open(dataset_path, "r") as fp:
        data = json.load(fp)

    X = np.array(data["mfcc"])
    y = np.array(data["labels"])

    return X, y
# def load_data2(dataset_ab):
#     with open(dataset_ab, "r") as fp:
#         data = json.load(fp)
#         X2 = np.array(data["mfcc"])
#         y2 = np.array(data["labels"])
#
#         return X2, y2

def prepare_dataset(test_size, validation_size):

    X, y = load_data(dataset_path)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)

    X_train = X_train[..., np.newaxis]  #As cnn takes 4d array and it is in 3d format

    X_validation = X_validation[..., np.newaxis] #4d shape (725, 87, 13, 1)
    X_test = X_test[..., np.newaxis]

    return X_train, X_validation, X_test, y_train, y_validation, y_test

def build_model(input_shape):

    model = keras.Sequential() #create model

    model.add(keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape))  #1st conv layer
    model.add(keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding="same"))
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape))  # 2nd conv layer
    model.add(keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding="same"))
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Conv2D(32, (2, 2), activation="relu", input_shape=input_shape))  # 3rd conv layer
    model.add(keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding="same"))
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Flatten()) #FLatten the layer and give it into dense layer
    model.add(keras.layers.Dense(64, activation="relu"))
    model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.Dense(3, activation="softmax")) #Output layer

    return model

def predict(model, X, y):
    X = X[np.newaxis, ...]
    prediction = model.predict(X)
    with open(dataset_path, "r") as fp:
        data = json.load(fp)
    print(prediction, data["mappings"][y])

    check = np.argmax(prediction)

    if(prediction[0,check]<0.60):
        print('Anomaly Detected')
    else:
        predicted_index = np.argmax(prediction, axis=1) #Output = 1d array
        print(predicted_index)
        pred = predicted_index[0]

        print("Expected class: {}, Predicted class: {}".format(data["mappings"][y], data["mappings"][pred]))



if __name__ == "__main__":
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_dataset(0.25,0.2)

    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
    model = build_model(input_shape) #build the network

    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"]) #compiled it

    model.fit(X_train, y_train, validation_data=(X_validation, y_validation), batch_size=32, epochs=30)  #train the model

    test_error, test_accuracy = model.evaluate(X_test, y_test, verbose = 1) #evaluate the cnn

    print("Accuracy on test set is: {}".format(test_accuracy))

    # X2, y2 = load_data2(dataset_ab)
    # X2 = X2[..., np.newaxis]
    X = X_test[210]
    y = y_test[210]
    # X = X2[200]
    # y = y2[200]
    predict(model, X, y)
