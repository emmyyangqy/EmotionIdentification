import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
import cv2
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

emotioncount = 7
epochnum = 2

def load_dataset():

    base_dir = 'data/FER-2013 - thresholds'
    
    # Create empty arrays to store the training and test data
    global x_train, y_train, x_test, y_test 
    x_train = []
    y_train = []
    x_test = []
    y_test = []

    # Get the list of categorical labels
    labels = os.listdir(os.path.join(base_dir, 'train'))
    num_labels = len(labels)

    # Loop through the training and test directories and load the images
    desired_shape = (100, 100, 3)
    for phase in ['train', 'test']:
        phase_dir = os.path.join(base_dir, phase)
        for label in os.listdir(phase_dir):
            label_dir = os.path.join(phase_dir, label)
            for image in os.listdir(label_dir):
                
                image_path = os.path.join(label_dir, image)
                image_data = cv2.imread(image_path, cv2.IMREAD_COLOR)
                image_data.resize(desired_shape)

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

    
    x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
    y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
    x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)
    y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)


    # Print the shapes of the training and test sets
    print(f'x_train shape: {x_train.shape}')
    print(f'y_train shape: {y_train.shape}')
    print(f'x_test shape: {x_test.shape}')
    print(f'y_test shape: {y_test.shape}')



def create_model():
    global model
    
    model = tf.keras.Sequential()
    #model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 3)))
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Dropout(0.5)) # dropout layer

    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Dropout(0.5)) 

    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Dropout(0.5))  

    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Dropout(0.5)) 

    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Dropout(0.5))  

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5)) 
    model.add(tf.keras.layers.Dense(emotioncount, activation='softmax'))  # Change the output layer for image classification

    return model 


def train_model(model, x_train, y_train, x_test, y_test):
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-1), metrics=['accuracy'])
    batch_size = 100
    num_epochs = 10
    global history

    for epoch in range(num_epochs):
        indices = tf.random.uniform(
            shape=(batch_size,), minval=0, maxval=x_train.shape[0], dtype=tf.int32)
        batch_x = tf.gather(x_train, indices)
        batch_y = tf.gather(y_train, indices)
        history = model.fit(batch_x, batch_y, epochs=1)

    test_loss, test_acc = model.evaluate(x_test, y_test)
    print('Test accuracy:', test_acc)
    print('Test loss:', test_loss)
    # Save the model
    model.save("CNN6model.h5")



load_dataset()
create_model()
train_model(model, x_train, y_train, x_test, y_test)
print(history)

train_loss=history.history['loss']
train_acc=history.history['accuracy']

epochs=range(len(train_acc))

plt.plot(train_acc,'r',label='train_acc')
plt.title('train_acc vs val_acc')
plt.legend()
plt.figure()
plt.show()


