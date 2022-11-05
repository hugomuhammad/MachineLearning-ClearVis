import tensorflow as tf
import os
import load_data
import model

#function for training the model
def train_model(model, epochs, train_generator, validation_generator):

  #write training history to csv format
  saved_path = os.path.join('training_history', 'model_training.log')
  csv_logger = tf.keras.callbacks.CSVLogger(saved_path, separator=',', append=False)
  
  return model.fit(
         train_generator,
         epochs=epochs, 
         validation_data=validation_generator,
         verbose=2,
         callbacks=[csv_logger])

#function for saving the model
def save_model(model, model_name):
  save_path = os.path.join('saved_model', model_name)
  model.save(save_path)
  return save_path

#function for loading the model
def load_model(model_path):
  return tf.keras.models.load_model(model_path)

if __name__ == '__main__':
    #load data
    train_dir, validation_dir, test_dir = load_data.load_data()
    train_generator, validation_generator = load_data.augment_data(train_dir, validation_dir)
    
    #create the model
    model = model.create_model()

    #train the model
    train_model = train_model(model, 20, train_generator, validation_generator)
    
    #save and load the model
    saved_path = save_model(model, 'cnnsvm_retinoblastoma_model.h5')
    load_model(saved_path)

        
        