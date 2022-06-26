# MachineLearning-ClearVis
Machine Learning Modelling Process for ClearVis Application

## About The Project 
This is Bangkit 2022 final project about Retinoblastoma detection app as one of the requirements for graduation.

What are doing here?
- Creating retinoblastoma detection dataset
- Build The CNN model
- Converting model to TFLite format

why we decided to make this model?
- Retinoblastoma is a malignancy of the eye that is found in the retina that happens mostly to young child that impacts the vision of children. Late Diagnosis and treatment of retinoblastoma in developing countries including Indonesia results in extraocular metastases, vision loss and death.

What is our objective?
- to provide an application that could perform an early detection of retinoblastoma disease

## Dataset
| Dataset | [Download Dataset](https://github.com/hugomuhammad/MachineLearning-ClearVis/blob/main/dataset/Retinoblastoma-Dataset.zip) |
| ------ | ------ |
| Title | Retinoblastoma dataset, ClearVis |
| Description | Retinoblastoma dataset consist of 2 classes normal eye and retinoblastoma eye. created for bangkit 2022 capstone project |
| Classes | Normal eye, Retinoblastoma eye |
| Records | 535 Train set, 50 Validation set, 40 Test set |
| Structure | <img src="https://github.com/hugomuhammad/MachineLearning-ClearVis/blob/main/assets/Screen%20Shot%202022-06-26%20at%2016.22.40.png"/> |
| Size | 21,2 MB |

Sample:
- Normal eye
<img src="https://github.com/hugomuhammad/MachineLearning-ClearVis/blob/main/assets/o-EYES-facebook.jpeg"/>
- Retinoblastoma eye
<img src="https://github.com/hugomuhammad/MachineLearning-ClearVis/blob/main/assets/Retinoblastoma_Kanan_2.jpg"/>

## Work steps
The model creation process goes through several steps, including:
1. Creating the dataset
We gathered a picture of normal eyes and retinoblastoma eyes from the internet with manual web scraping, then structure it into directory for each set of data and classes as can be seen in the dataset documentation above. 

2. Image augmentation
We use image augmentation technique to artificially expand the data-set. This is helpful since  we only had a data-set with very few data samples. In case of Deep Learning, this situation is bad as the model tends to over-fit when we train it on limited number of data samples.

3. Create the model architecture
we use trasnfer learning vgg16 pre-trained model on imagenet to save time and resources from having to train multiple machine learning models from scratch then we add fully connected layer before the output layer.

4. Train the model
we trained the model with training set and validation set that has been augmented before.

5. Evaluate the model
After the training is done, we evaluate the model with the testing set. we check the accuracy and confussion matrix  (true positive, true negative, false positive and false negative).

6. Saving the model
After everything is done we saved the model to the .h5 file format.

7. Convert to TFLite format
finallly we convert the .h5 model to .tflite format for the android deployment using tensorflow lite.

Full code implementation and results can be seen on [Retinoblastoma_detection.ipynb](https://github.com/hugomuhammad/MachineLearning-ClearVis/blob/main/Retinoblastoma_Detection.ipynb)

## Conclusion
In coclusion we are able to create classification model for retinoblastoma case with 78% accuracy on test data with realitively small amount of data by using transfer learning. for the future development we are aiming to get more data and trying to reduce false negative and increase true negative rate.

## Reference
- [Accurate leukocoria predictor based on deep VGG-net CNN technique](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/iet-ipr.2018.6656)
- [Early Prediction and Diagnosis of Retinoblastoma Using Deep Learning Techniques](https://arxiv.org/abs/2103.07622)
- [Eye disorder dataset classifier with VGG16](https://www.kaggle.com/code/chetbounl/eye-disorder-dataset-classifier-with-vgg16/notebook)
- [GRADLE](https://play.google.com/store/apps/details?id=net.leuko.leuko_android&hl=in&gl=US)

## How to Run The Notebook
### in google colab
1. Open Retinoblastoma_Detection.ipynb in google colab
2. Change runtime type using GPU
3. Upload dataset from dataset folder in this repository
4. Run each cell on the notebook

### in local machine
Install all the dependencies 
- Install Tensorflow:
```sh
pip install tensorflow
```
- Install Numpy:
 ```sh 
pip install numpy
  ```
- Install matplotlib:
```sh
pip install matplotlib
```
Run the notebook
1. Open visual studio code 
2. Install jupyter notebook extension
3. Add project folder to workspace
4. Run each cell

## Build With
This application model build with some technology, whice is:
- Python 3.9
- Jupyter Notebook
- Tensorflow 2.5
- numpy 1.19.5
- Google Colab








