import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

# Load the saved model
model = tf.keras.models.load_model('m2.keras', custom_objects={'silu': lambda x: x * tf.math.sigmoid(x)})

# Constants
IMG_SIZE = (256, 256)

# Function to load and preprocess a single image
def load_and_preprocess_image(image_path):
    img = load_img(image_path, target_size=IMG_SIZE)  # Resize the image
    img_array = img_to_array(img) / 255.0  # Convert to array and normalize
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions for model input
    return img_array

# Provide the path to the test image
image_path = 'download.jpg'  # Replace with your image path

# Load and preprocess the image
test_image = load_and_preprocess_image(image_path)

# Predict using the model
predicted_output = model.predict(test_image)

# Display the original and predicted images side by side
def display_images(original_img_path, predicted_img_array):
    original_img = load_img(original_img_path, target_size=IMG_SIZE)
    predicted_img = predicted_img_array[0]  # Remove batch dimension

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # Display original image
    axs[0].imshow(original_img)
    axs[0].set_title('Original Image')
    axs[0].axis('off')

    # Display predicted image
    axs[1].imshow(predicted_img)
    axs[1].set_title('Predicted Output')
    axs[1].axis('off')

    plt.show()

# Display the images
display_images(image_path, predicted_output)
