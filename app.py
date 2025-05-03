import streamlit as st
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Title
st.title("Art Style Recognition 🎨")
st.write("Upload an image to predict the art style!")

# Load model
model = tf.keras.models.load_model('my_art_style_model.h5')

# Art style classes
art_styles = [
    "Abstract Expressionism",
    "Art Nouveau Modern",
    "Baroque",
    "Cubism",
    "Expressionism",
    "Impressionism",
    "Naive Art Primitivism",
    "Northern Renaissance",
    "Post Impressionism",
    "Realism",
    "Rococo",
    "Surrealism",
    "Symbolism"
]

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    # Preprocess image
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction, axis=1)[0]
    predicted_label = art_styles[predicted_index]

    # Add text overlay on image
    draw = ImageDraw.Draw(image)
    text = f"Predicted Art Style: {predicted_label}"
    font = ImageFont.load_default()  # You can customize the font size here
    text_position = (10, 10)  # Position at the top-left corner

    # Draw the text over the image
    draw.text(text_position, text, font=font, fill="white")

    # Show the image with the text
    st.image(image, caption=f'Uploaded Image with Predicted Style: {predicted_label}', use_container_width=True)

    st.success(f"🎨 Predicted Art Style: **{predicted_label}**")
