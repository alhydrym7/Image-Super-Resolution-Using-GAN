# import streamlit as st
# import os
# import tempfile
# from PIL import Image
# import numpy as np
# from keras.models import load_model
# import Utils, Utils_model
# from Utils_model import VGG_LOSS

# # Function to save uploaded files to a temporary directory
# def save_uploaded_files(uploaded_files):
#     temp_dir = tempfile.mkdtemp()
#     for uploaded_file in uploaded_files:
#         if uploaded_file is not None:
#             file_path = os.path.join(temp_dir, uploaded_file.name)
#             with open(file_path, "wb") as f:
#                 f.write(uploaded_file.getbuffer())
#     return temp_dir

# # Function to preprocess and resize images
# def preprocess_images(uploaded_files, target_size=(96, 96)):
#     processed_images = []
#     for uploaded_file in uploaded_files:
#         image = Image.open(uploaded_file)
#         image = image.resize(target_size, Image.BICUBIC)
#         processed_images.append(np.array(image))
#     return processed_images

# st.title("Image Super-Resolution Tester")

# # Upload image or images
# uploaded_files = st.file_uploader("Choose images", accept_multiple_files=True)

# # Load the model
# model_path = st.text_input("Enter the path to the model:", './model/gen_model3000.h5')
# loss = VGG_LOSS((96, 96, 3))
# model = load_model(model_path, custom_objects={'vgg_loss': loss.vgg_loss})

# output_dir = "./output_v2/"
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)

# if uploaded_files and model_path:
#     temp_dir = save_uploaded_files(uploaded_files)

#     # Preprocess and resize images
#     preprocessed_images = preprocess_images(uploaded_files)

#     # Convert preprocessed images to NumPy array
#     x_test_lr = np.array(preprocessed_images)
#     x_test_lr = Utils.normalize(x_test_lr)  # Normalize if your model requires it

#     # Generate and save output images
#     gen_images = model.predict(x_test_lr)
#     gen_images = Utils.denormalize(gen_images)  # Denormalize if you did normalization

#     for i, img in enumerate(gen_images):
#         # Save each generated image
#         img = Image.fromarray(img)
#         img.save(os.path.join(output_dir, f"output_{i}.png"))

#         # Display the uploaded image and the output image
#         st.image(uploaded_files[i], caption='Original Image', use_column_width=True)
#         st.image(os.path.join(output_dir, f"output_{i}.png"), caption='Super-Resolved Image', use_column_width=True)









import streamlit as st
import os
import tempfile
from PIL import Image
import numpy as np
from keras.models import load_model
import Utils, Utils_model
from Utils_model import VGG_LOSS

# Function to save uploaded files to a temporary directory
def save_uploaded_files(uploaded_files):
    temp_dir = tempfile.mkdtemp()
    for uploaded_file in uploaded_files:
        if uploaded_file is not None:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
    return temp_dir

# Function to preprocess and resize images
def preprocess_images(uploaded_files, target_size=(96, 96)):
    processed_images = []
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        image = image.resize(target_size, Image.BICUBIC)
        processed_images.append(np.array(image))
    return processed_images

def super_resolution_page():
    st.title("Image Super-Resolution Generator")

    # Image uploader in sidebar
    uploaded_files = st.sidebar.file_uploader("Upload images for super-resolution:", accept_multiple_files=True, type=["jpg", "jpeg", "png"])
    
    # Load the model
    model_path = st.sidebar.text_input("Enter the path to the model:", './model/gen_model3000.h5')
    loss = VGG_LOSS((96, 96, 3))
    model = load_model(model_path, custom_objects={'vgg_loss': loss.vgg_loss})

    output_dir = "./output/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if uploaded_files and model_path:
        temp_dir = save_uploaded_files(uploaded_files)
        preprocessed_images = preprocess_images(uploaded_files)
        x_test_lr = np.array(preprocessed_images)
        x_test_lr = Utils.normalize(x_test_lr)  # Normalize if your model requires it

        # Generate and save output images
        st.sidebar.text("Generating super-resolution images...")
        gen_images = model.predict(x_test_lr)
        gen_images = Utils.denormalize(gen_images)  # Denormalize if you did normalization

        for i, img in enumerate(gen_images):
            # Save each generated image
            img = Image.fromarray(img)
            img.save(os.path.join(output_dir, f"output_{i}.png"))

            # Display the uploaded image and the output image
            st.image(uploaded_files[i], caption='Original Image', use_column_width=True)
            st.image(os.path.join(output_dir, f"output_{i}.png"), caption='Super-Resolved Image', use_column_width=True)
        st.sidebar.text("Super-resolution images generated!")

# def home_page():
#     st.title("Welcome to the Image Super-Resolution App")
#     st.write("This application enhances the resolution of images using a deep learning model.")
#     st.write("Navigate to the Super-Resolution Generator page to upload and enhance your images.")

def home_page():
    st.title("Welcome to the:")
    st.title("Image Super-Resolution App")

    st.header("Elevate Your Images with Cutting-Edge AI")
    st.write("""
        Step into the future of image processing with our Super-Resolution App, 
        leveraging advanced deep learning techniques to transform your low-resolution images into 
        high-definition masterpieces. Our sophisticated neural networks have been finely tuned to 
        upscale images with incredible detail and clarity, breathing new life into every pixel.
        Whether you're a photographer looking to refine your shots, a designer in pursuit of 
        perfection, or just want to enhance old family photos, our app is your gateway to visual excellence.
        Ready to unlock the full potential of your visuals? Head over to the Super-Resolution Generator 
        page, where you can effortlessly upload and magnify your images with unparalleled quality.
    """)


# Sidebar with project information and navigation
st.sidebar.markdown("### Project Information")
st.sidebar.markdown("#### Developer:")
st.sidebar.markdown("Mohammed Mufleh")
st.sidebar.markdown("#### University Number:")
st.sidebar.markdown("4010356")
st.sidebar.markdown("#### Under Supervision of:")
st.sidebar.markdown("Dr. Ahmed Elhayek")

# Page navigation
page = st.sidebar.radio('Navigate to', ['Home', 'Super-Resolution Generator'])

if page == 'Home':
    home_page()
elif page == 'Super-Resolution Generator':
    super_resolution_page()
