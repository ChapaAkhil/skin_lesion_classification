import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization

def create_model():
    model = Sequential()

    model.add(Conv2D(16, kernel_size=(3,3), input_shape=(28,28,3), activation='relu', padding='same'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(BatchNormalization())

    model.add(Conv2D(32, (3,3), activation='relu'))
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(BatchNormalization())

    model.add(Conv2D(128, (3,3), activation='relu'))
    model.add(Conv2D(256, (3,3), activation='relu'))

    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())

    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(7, activation='softmax'))

    return model
