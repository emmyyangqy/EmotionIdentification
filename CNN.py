import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
import cv2

def load_dataset():
    # Set the base directory for the images
    base_dir = 'data/CK+_withtesttrain/'
    global x_train, y_train, x_test, y_test 
    # Create empty arrays to store the training and test data
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    # Get the list of categorical labels
    labels = os.listdir(os.path.join(base_dir, 'train'))
    num_labels = len(labels)
    
    # Loop through the training and test directories and load the images
    for phase in ['train', 'test']:
        phase_dir = os.path.join(base_dir, phase)
        for label in os.listdir(phase_dir):
            label_dir = os.path.join(phase_dir, label)
            for image in os.listdir(label_dir):
                image_path = os.path.join(label_dir, image)
                image_data = cv2.imread(image_path, cv2.IMREAD_COLOR)
                if phase == 'train':
                    x_train.append(image_data)
                    y_train.append(label)
                else:
                    x_test.append(image_data)
                    y_test.append(label)
    # Convert the lists to NumPy arrays
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    # Encode the labels as integers
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train)
    y_test = encoder.transform(y_test)
    
    # Reshape the labels to have the same shape as the number of categorical subfolders
    y_train = np.resize(y_train, (y_train.shape[0], num_labels))
    y_test = np.resize(y_test, (y_test.shape[0], num_labels))

    # Print the shapes of the training and test sets
    print(f'x_train shape: {x_train.shape}')
    print(f'y_train shape: {y_train.shape}')
    print(f'x_test shape: {x_test.shape}')
    print(f'y_test shape: {y_test.shape}')

def create_model():
    global model

    # Build the model
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 3)))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Dropout(0.5))  # Add a dropout layer
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Dropout(0.5))  # Add a dropout layer
    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Dropout(0.5))  # Add a dropout layer
    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Dropout(0.5))  # Add a dropout layer
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))  # Add a dropout layer
    model.add(tf.keras.layers.Dense(7, activation='softmax'))  # Change the output layer for image classification
    return model
    
def train_model(model, x_train, y_train, x_test, y_test):
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=64)
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    
load_dataset()
create_model()
train_model(model, x_train, y_train, x_test, y_test)
