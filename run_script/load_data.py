import pandas as pd
import tensorflow as tf
import os
import numpy as np
from zipfile import ZipFile


#function for loading the data from zip file
def load_data():
    os.system('unzip -o dataset/Retinoblastoma-Dataset.zip -d dataset/')
    
   
    # Current directory
    cwd = os.getcwd()

    # base directory for dataset
    base_dir = os.path.join(cwd, 'dataset/Retinoblastoma-dataset/')

    train_dir = os.path.join(base_dir, 'training') # training set directory
    validation_dir = os.path.join(base_dir, 'validation') # validation set directory
    test_dir = os.path.join(base_dir, 'testing') # testing set directory

    return train_dir, validation_dir, test_dir


#function for augmenting the data
def augment_data(train_dir, validation_dir):
    # set image size

    img_height = 224
    img_width = 224

    # create data generator
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                    rescale=1./255,
                    rotation_range=20,
                    horizontal_flip=True,
                    vertical_flip=True,
                    shear_range = 0.2,
                    fill_mode = 'nearest')

    validation_datagen =  tf.keras.preprocessing.image.ImageDataGenerator(
                    rescale=1./255,
                    rotation_range=20,
                    horizontal_flip=True,
                    vertical_flip=True,
                    shear_range = 0.2,
                    fill_mode = 'nearest')

    # create data generator for training set
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_height, img_width),
        batch_size=32,
        class_mode='categorical')

    # create data generator for validation set
    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(img_height, img_width),
        batch_size=32,
        class_mode='categorical')


    return train_generator, validation_generator


if __name__ == '__main__':
    #load augmented data
    train_dir, validation_dir, test_dir = load_data()
    train_generator, validation_generator = augment_data(train_dir, validation_dir)
    
    #print(train_generator.class_indices)
    

