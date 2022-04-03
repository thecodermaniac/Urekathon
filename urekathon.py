import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import numpy as np
import streamlit as st

model_cancer = tf.keras.models.load_model(
    r'C:\Users\sasuk\Documents\urekathon\BreastCancerSegmentor.h5')
model_cancer_class = tf.keras.models.load_model(
    r'C:\Users\sasuk\Documents\urekathon\classifier_cancer_breast.h5')

class_label = {0: 'malignant', 1: 'benign'}


def image_segmenter(imgpath):
    img = plt.imread(imgpath)
    img = cv2.resize(img, (256, 256))
    img = np.expand_dims(img, axis=0)
    affected_tissue = model_cancer.predict(np.array(img))
    class_cancer = model_cancer_class.predict(np.array(img))
    return affected_tissue, class_cancer


def to_raw(string):
    return fr"{string}"


st.header("Breast Cancer Image Segmentor")
st.subheader('This web app will take ultrasonography images of breast tissue and mark the tumorous tissue and type of breast cancer')
img_path = ''
img_path = st.text_input("Give the full path of Image file")
if len(img_path) != 0:
    img_path = to_raw(img_path)
    segmented, cls = image_segmenter(img_path)
    img = plt.imread(img_path)
    img = cv2.resize(img, (256, 256))

    # temp_img = plt.imshow(segmented[0][:, :, 0])
    st.text("masked image")
    st.image(segmented[0][:, :, 0])
    st.write("real image")
    st.image(img)

    st.title(f'Its a {class_label[cls.argmax()]} breast cancer')


else:
    st.write('Please enter a valid Image path')
# print(class_label[cls.argmax()])
