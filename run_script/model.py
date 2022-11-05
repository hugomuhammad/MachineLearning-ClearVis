#import tensorflow as tf
import tensorflow as tf
import os

def create_model():

    #setting parameters for svm classification layer
    kernel_regularizer = tf.keras.regularizers.l2(0.01)
    activation_funct = 'softmax' 
    loss = 'squared_hinge'
    
    #create model  
    model = tf.keras.models.Sequential([
      tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(224, 224, 3)),
      tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
      tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
      tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
      tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
      tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
      tf.keras.layers.Conv2D(512, (3, 3), activation='relu'),
      tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
      tf.keras.layers.Conv2D(512, (3, 3), activation='relu'),
      tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(512, activation='relu'),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(256, activation='relu'),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(2, kernel_regularizer = kernel_regularizer, activation=activation_funct)
      ])

  #compile model 
    model.compile(loss = loss, 
                optimizer=tf.optimizers.Adam(),
                metrics=['accuracy'])
  
  #return model
    return model

if __name__ == '__main__':
    model = create_model()
    print(model.summary())

        
        