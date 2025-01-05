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

# Streamlit UI
st.title("Fourier Transform with Streamlit")
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    img_array = np.array(img)
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    fft_images, fft_images_log = rgb_fft(img_array)

    # Display FFT for each color channel
    st.subheader("Fourier Transform (Frequency Domain)")
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(fft_images_log[0], cmap='gray')
    ax[0].set_title("Red Channel in Frequency Domain")
    ax[1].imshow(fft_images_log[1], cmap='gray')
    ax[1].set_title("Green Channel in Frequency Domain")
    ax[2].imshow(fft_images_log[2], cmap='gray')
    ax[2].set_title("Blue Channel in Frequency Domain")
    st.pyplot(fig)

    # Mask radius slider
    radius = st.slider("Select Mask Radius", min_value=10, max_value=200, value=50)

    # Create a mask and apply it
    mask = np.zeros_like(fft_images[0], dtype='uint8')
    rows, cols = mask.shape
    cv2.circle(mask, (cols // 2, rows // 2), radius, 255, -1)
    
    result_images = []
    for i in range(3):
        masked_fft = fft_images[i] * (mask / 255)
        result_images.append(masked_fft)

    transformed = inverse_fourier(result_images)
    transformed_clipped = np.clip(transformed, 0, 255).astype('uint8')

    # Display the inverse transformed image
    st.subheader("Inverse Fourier Transformed Image")
    st.image(transformed_clipped, caption="Reconstructed Image", use_column_width=True)
