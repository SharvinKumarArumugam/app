import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Function to apply mask and perform inverse Fourier transform
def inverse_fourier(image):
    final_image = []
    for c in image:
        channel = abs(np.fft.ifft2(c))
        final_image.append(channel)
    final_image_assembled = np.dstack([
        final_image[0].astype('int'),
        final_image[1].astype('int'),
        final_image[2].astype('int')
    ])
    return final_image_assembled

def normalize_image(img):
    img = img / np.max(img)
    return (img * 255).astype('uint8')

def rgb_fft(image):
    fft_images = []
    fft_images_log = []
    for i in range(3):
        rgb_fft = np.fft.fftshift(np.fft.fft2(image[:, :, i]))
        fft_images.append(rgb_fft)
        fft_images_log.append(np.log(abs(rgb_fft) + 1))
    return fft_images, fft_images_log

# Streamlit app layout
st.title("Interactive Fourier Transform Visualization")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = np.array(Image.open(uploaded_file))

    st.image(img, caption="Uploaded Image", use_column_width=True)

    fft_images, fft_images_log = rgb_fft(img)

    st.write("### RGB Channels in Frequency Domain")
    col1, col2, col3 = st.columns(3)
    col1.image(fft_images_log[0], caption="Red Channel", use_column_width=True, clamp=True)
    col2.image(fft_images_log[1], caption="Green Channel", use_column_width=True, clamp=True)
    col3.image(fft_images_log[2], caption="Blue Channel", use_column_width=True, clamp=True)

    # Slider for mask radius
    radius = st.slider("Select Mask Radius", min_value=10, max_value=200, value=50, step=10)

    # Create and apply mask
    mask = np.zeros_like(fft_images[0])
    rows, cols = mask.shape
    cv2.circle(mask, (cols // 2, rows // 2), radius, 255, -1)
    
    result_images = [fft * (mask / 255) for fft in fft_images]

    transformed = inverse_fourier(result_images)
    transformed_clipped = normalize_image(transformed)

    st.write("### Inverse Fourier Transformed Image")
    st.image(transformed_clipped, caption="Reconstructed Image", use_column_width=True)
