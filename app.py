import os
import streamlit as st
import cv2
import numpy as np
import joblib
from skimage.feature import hog

# Load the trained model and label encoder
model = joblib.load("lung_cancer_hard_voting.pkl")
encoder = joblib.load("label_encoder.pkl")

# Constants
IMAGE_SIZE = (224, 224)
SAMPLE_IMAGE_DIR = "sample images"  # Folder name with 4 sample images
LABELS = {
    0: "⚠️ Benign tumor detected. Regular monitoring is advised. Consult your doctor for further evaluation.",
    1: "✅ No immediate signs of lung cancer detected. However, regular medical check-ups and a healthy lifestyle are essential. If any concerns arise, consult a doctor for further evaluation.",
    2: "✅ No signs of lung cancer detected. Maintain a healthy lifestyle and regular check-ups."
}

def preprocess_image(image):
    """Preprocess the uploaded image."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    image = cv2.resize(image, IMAGE_SIZE)  # Resize to match training size
    image = image / 255.0  # Normalize pixel values
    return image

def extract_features(image):
    """Extract HOG features."""
    features, _ = hog(image, pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=True)
    return np.array(features).reshape(1, -1)  # Reshape for model input

def predict(image):
    """Make a prediction based on image input."""
    preprocessed = preprocess_image(image)
    features = extract_features(preprocessed)
    prediction = model.predict(features)
    predicted_label = encoder.inverse_transform([prediction[0]])[0]
    return LABELS[int(predicted_label)]

# Streamlit UI
st.title("🩺 Lung Cancer Detection")
st.write("Upload a lung CT scan image or choose a sample image to get a prediction.")

# --- Upload your own image ---
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

# --- Choose a sample image ---
sample_images = [f for f in os.listdir(SAMPLE_IMAGE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
selected_sample = st.selectbox("Or choose a sample image", ["None"] + sample_images)

image = None

# Load uploaded image
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.image(image, caption="Uploaded Image", use_column_width=True)

# Load selected sample image
elif selected_sample != "None":
    image_path = os.path.join(SAMPLE_IMAGE_DIR, selected_sample)
    image = cv2.imread(image_path)
    st.image(image, caption=f"Sample Image: {selected_sample}", use_column_width=True)

# Predict if image is loaded
if image is not None:
    if st.button("Predict"):
        result = predict(image)
        st.success(f"Prediction: **{result}**")
