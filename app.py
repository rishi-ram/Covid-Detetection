from numpy.core.fromnumeric import resize
import streamlit as st
import pickle
import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np
from PIL import Image,ImageOps
classes=['Covid','Pleural Effusion','Normal CT']
st.write("""
         # Covid and Pleural Effusion Detection from CT Scan image
         """
         )
st.write("")

file = st.file_uploader("Please upload an image file", type=["jpeg","jpg", "png"])

if file is None:
    st.write("Upload CT Image")
else:
    image_pil=Image.open(file)
    image = ImageOps.fit(image_pil, (244,244), Image.ANTIALIAS)
    array_img=np.asarray(image)
    resized_image=cv2.resize(array_img,(244,244))
    np_array=np.array(resized_image)
    np_array=np_array/255
    img_reshape = np_array[np.newaxis,...]
    model = keras.models.load_model("covid_model")
    pred1=model.predict(img_reshape)
    # st.write("Preddddd",np.argmax(pred1))
    index=np.argmax(pred1)
    percentage=pred1[0][index]
    st.image(img_reshape,use_column_width=True)
    st.write("""
            # {0} Detected {1:.2f}% 
            """.format(classes[index],percentage*100)
            )

