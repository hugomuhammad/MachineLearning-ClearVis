import streamlit as st
import tensorflow as tf
import streamlit as st
import cv2
from PIL import Image, ImageOps
import numpy as np


@st.cache(allow_output_mutation=True)

def load_model():
  model=tf.keras.models.load_model('Saved_model/cnnsvm_retinoblastoma_model.h5')
  return model

def predict(image_data, model):
    
        size = (224,224)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #img_resize = (cv2.resize(img, dsize=(75, 75),    interpolation=cv2.INTER_CUBIC))/255.
        
        img_reshape = img[np.newaxis,...]
    
        prediction = model.predict(img_reshape)
        
        return prediction


with st.spinner('Model is being loaded..'):
  model=load_model()

st.write("""
         # Retinoblastoma Classification App
         """
         )

st.write('## Get Started')
st.write('1. Upload an eye photos from flash photography like the example below')
image = Image.open('close-asian-woman-eyes-flash-260nw-433717459.jpg')
st.image(image, use_column_width='auto')
st.write('2. Prediction result will be shown immediately')

st.write('## Upload image file below')
file = st.file_uploader("", type=["jpg", "png", "jpeg"])

st.set_option('deprecation.showfileUploaderEncoding', False)

if file is None:
    pass

else:
    image = Image.open(file)
    st.image(image, use_column_width='auto')
    predictions = predict(image, model)

    if round(float(predictions[0][0])) == 1:
        results = 'normal'
        print("user's eyes is normal")
        print(predictions)
        st.write("user's eyes is normal")
    elif round(float(predictions[0][1])) == 1:
        results = 'retinoblastoma'
        print("user's eyes is retinoblastoma")
        print(predictions)
        st.write("user's eyes is suspected with retinoblastoma")
