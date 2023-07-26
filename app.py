import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model


# Sidebar - Model Selection
st.sidebar.header('Choose your model')
model_name = st.sidebar.selectbox('Model selection', ['inceptionGRU.h5', 'inceptionLSTM.h5'])

# Load model
model = load_model(model_name)

# Main - Image Upload and Captioning
st.header('Image Captioning App')
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    if st.button('Generate Caption'):
        # Here you would use your model to generate a caption, for instance:
        caption = model.predict(image)
        st.write(caption)

