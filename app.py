import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

model = load_model("model.h5")  
with open("label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)



def get_predictions(uploaded_image):

    test_img = uploaded_image.resize((224, 224))
    test_img_array = image.img_to_array(test_img)
    test_img_array = test_img_array / 255.0
    test_img_array = np.expand_dims(test_img_array, axis=0)

    predictions = model.predict(test_img_array)

    articleType_encoder = label_encoders['articleType']
    baseColour_encoder = label_encoders['baseColour']
    gender_encoder = label_encoders['gender']
    season_encoder = label_encoders['season']

    articleType_pred = np.argmax(predictions[0])
    baseColour_pred = np.argmax(predictions[1])
    gender_pred = np.argmax(predictions[2])
    season_pred = np.argmax(predictions[3])

    articleType_label = articleType_encoder.inverse_transform([articleType_pred])[0]
    baseColour_label = baseColour_encoder.inverse_transform([baseColour_pred])[0]
    gender_label = gender_encoder.inverse_transform([gender_pred])[0]
    season_label = season_encoder.inverse_transform([season_pred])[0]

    return articleType_label, baseColour_label, gender_label, season_label

st.title('Fashion Attribute Classifier')
uploaded_image = st.file_uploader('Upload an Image', type=['png', 'jpg', 'jpeg'])

if uploaded_image:
    img = Image.open(uploaded_image)
    if st.button('Submit'):
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(img, caption='Uploaded Image', width=300)

        with col2:
            st.markdown("### Predictions")
            articleType, baseColour, gender, season = get_predictions(img)

            st.write(f'**Article Type:** {articleType}')
            st.write(f'**Base Colour:** {baseColour}')
            st.write(f'**Gender:** {gender}')
            st.write(f'**Season:** {season}')
