import streamlit as st
import os
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
import numpy as np
import pickle as pk
from mtcnn import MTCNN
from sklearn.metrics.pairwise import cosine_similarity
import cv2
from PIL import Image

st.title("Which bollywood celebrity are you?")

uploaded_image = st.file_uploader('Choose an image')

model = VGGFace(model='resnet50', include_top=False, input_shape=(224,224,3), pooling='avg')
detector = MTCNN()
feature_list = pk.load(open('embeddings.pkl', 'rb'))
filenames = pk.load(open('filenames.pkl', 'rb'))
def save_uploaded_file(uploaded_image):
    try:
        with open(os.path.join('uploads', uploaded_image.name), 'wb') as f:
            f.write(uploaded_image.getbuffer())
        return True
    except:
        return False

def extract_featres(img_path):
    img = cv2.imread(img_path)
    orig_h, orig_w, channels = img.shape
    results = detector.detect_faces(img)
    x, y, width, height = 0, 0, orig_w, orig_h
    if (len(results)):
        x, y, width, height = results[0]['box']
    if (x<0):
        x = 0
    if (y<0):
        y = 0
    if (x + width > orig_w):
        width = orig_w - x
    if (y + height > orig_h):
        height = orig_h - y
    face = img[y:y + height, x:x + width]

    image = Image.fromarray(face)
    image = image.resize((224, 224))
    face_array = np.asarray(image).astype('float32')

    expanded_img = np.expand_dims(face_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img)
    result = model.predict(preprocessed_img).flatten()
    return result

def recommend(features):
    similarity = []
    for i in range(len(feature_list)):
        similarity.append(cosine_similarity(features.reshape(1, -1), feature_list[i].reshape(1,-1))[0][0])
    index_pos = sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[0][0]
    return index_pos

if uploaded_image is not None:
    if save_uploaded_file(uploaded_image):
        display_image = Image.open(uploaded_image)
        # st.image(display_image)
        st.text(os.path.join('uploads', uploaded_image.name))
        features = extract_featres(os.path.join('uploads', uploaded_image.name))
        index_pos = recommend(features)
        predicted_actor = " ".join(filenames[index_pos].split('\\')[1].split('_'))

        col1, col2 = st.columns(2)

        with col1:
            st.header('Your uploaded image')
            st.image(display_image)
        with col2:
            st.header("Seems like " + predicted_actor)
            st.image(filenames[index_pos],width=200)