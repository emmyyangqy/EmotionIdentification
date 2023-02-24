import numpy as np
import cv2
import os
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

emotioncount = 7
epochnum = 2

def load_dataset():

    # Set the base directory for the images
    base_dir = 'data/CK+ - L0.95 U100/'

    # Create empty arrays to store the training and test data   
    global new_data, labels 
    new_data = []
    labels = []

    # Get the list of categorical labels
    num_labels = 7

    desired_shape = (100, 100, 3)
    

    # Loop through the training and test directories and load the images
    for label in os.listdir(base_dir):
        label_dir = os.path.join(base_dir, label)
        for image in os.listdir(label_dir):
            
            image_path = os.path.join(label_dir, image)
            image_data = cv2.imread(image_path, cv2.IMREAD_COLOR)
            image_data.resize(desired_shape)

            new_data.append(image_data)
            labels.append(label)


    # Convert lists to NumPy arrays
    new_data = np.array(new_data)
    labels = np.array(labels)

    # Encode labels as integers
    encoder = LabelEncoder()
    labels = encoder.fit_transform(labels)

    # Reshape the labels to have the same shape as the number of categorical subfolders
    labels = np.resize(labels, (labels.shape[0], num_labels))
    
    new_data = tf.convert_to_tensor(new_data, dtype=tf.float32)
    labels = tf.convert_to_tensor(labels, dtype=tf.float32)

    # Print the shapes of the training and test sets
    print(f'new_data shape: {new_data.shape}')
    print(f'labels shape: {labels.shape}')

load_dataset()

model = tf.keras.models.load_model("CNN6model.h5")
loss, accuracy = model.evaluate(new_data, labels, verbose=0)

print('Test loss:', loss)
print('Test accuracy:', accuracy)
