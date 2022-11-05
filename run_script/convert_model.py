import tensorflow as tf
import os
import load_data
import model
import train
import pathlib

def convert(model):

  #Convert the model.
  converter = tf.lite.TFLiteConverter.from_keras_model(model) # path to the SavedModel directory
  converter.optimizations = [tf.lite.Optimize.DEFAULT]
  # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
  tflite_model = converter.convert()

  # Save the model.
  with open('Saved_model/cnnsvm_retinoblastoma_model.tflite', 'wb') as f:
    f.write(tflite_model)

if __name__ == '__main__':
  #load model
  model = train.load_model('Saved_model/cnnsvm_retinoblastoma_model.h5')
  #convert model
  convert(model)
        
        