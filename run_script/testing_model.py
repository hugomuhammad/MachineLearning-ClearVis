import tensorflow as tf
import numpy as np
import os
import load_data
import model
import train

#Function for predicting the class of the image
def predicted(model, target_directory, target):
  
  normal = 0
  retinoblastoma = 0

  try:
    #iterate through the retinoblastoma test directory
    dir = os.path.join(target_directory, target)
    for images_path in os.listdir(dir):
        path = images_path

        if path == '.DS_Store' or path.endswith('.heic'):
            continue

        #load image
        img = tf.keras.preprocessing.image.load_img(os.path.join(dir, path), target_size=(224,224))
        x =  tf.keras.preprocessing.image.img_to_array(img)
        x = np.expand_dims(x, axis=0)

        #predict image
        images = np.vstack([x])
        classes = model.predict(images, batch_size=10)
        
        if round(float(classes[0][0])) == 1:
          normal += 1
        elif round(float(classes[0][1])) == 1:
          retinoblastoma += 1
        
    #print total case and predicted case
    total_case = retinoblastoma + normal
    predicted_normal = normal
    predicted_retinoblastoma = retinoblastoma

    if target == 'normal':
      #print true negative, and false positive rate
      true_negative = predicted_normal
      false_positive = predicted_retinoblastoma
      return true_negative, false_positive
    
    if target == 'retinoblastoma':
      #print true negative, and false positive rate
      true_positive = predicted_retinoblastoma
      false_negative = predicted_normal
      return true_positive, false_negative
 
  except:
    pass

#Function for calculating model  metrics
def metrics(model, target_directory):
  TN, FP = predicted(model, target_directory, 'normal')
  TP, FN = predicted(model, target_directory, 'retinoblastoma')

  #calculating recall
  Recall = TP / (TP + FN) * 100
  print("\nRecall: {:.1f}%".format(Recall))

  #calculating precision
  Precision = TP/(TP + FP) * 100
  print("Precision: {:.1f}%".format(Precision))

  #calculating accuracy
  Accuracy = (TP + TN) / (TP+TN+FP+FN) * 100
  print("Accuracy: {:.1f}%".format(Accuracy))

  #calculating F1 score
  F1_score = (2 * Precision * Recall) / (Precision + Recall)
  print("F1 score: {:.1f}%".format(F1_score))


if __name__ == '__main__':
  #load model
  model = train.load_model('Saved_model/cnnsvm_retinoblastoma_model.h5')

  #load test data
  train_dir, validation_dir, test_dir = load_data.load_data()

  #evaluate model prediction result
  #evaluate testing data in normal directory
  predicted_normal_normal_dir, predicted_retinoblastoma_normal_dir = predicted(model, test_dir, 'normal')
  #evaluate testing data in retinoblastoma directory
  predicted_retinoblastoma_retinoblastoma_dir, predicted_normal_retinoblastoma_dir = predicted(model, test_dir, 'retinoblastoma')
  
  #get metrics
  metrics(model, test_dir)

  #print prediction results
  #in normal eye directory
  print('\nTotal normal case: ', predicted_normal_normal_dir + predicted_retinoblastoma_normal_dir)
  print('Total case predicted correctly: ', predicted_normal_normal_dir)
  print('Total case predicted incorrectly: ', predicted_retinoblastoma_normal_dir)
  print('')
  #in retinoblastoma eye directory
  print('Total retinoblasma case: ', predicted_normal_retinoblastoma_dir + predicted_retinoblastoma_retinoblastoma_dir)
  print('Total case predicted correctly: ', predicted_retinoblastoma_retinoblastoma_dir)
  print('Total case predicted incorrectly: ', predicted_normal_retinoblastoma_dir)

        
        