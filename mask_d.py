import os
import tensorflow as tf
import numpy as np
from PIL import Image

# Function to load and preprocess images
def load_and_preprocess_image(image_path):
    image = Image.open(image_path).resize((50, 50))  # Resize to 50x50
    image = np.array(image) / 255.0  # Normalize pixel values
    if image.shape[-1] != 3:  # Convert grayscale images to RGB if needed
        image = np.stack([image] * 3, axis=-1)
    return image

# Load dataset
mask_image_dataset_url = "./DataSet/Mask"
no_mask_image_dataset_url = "./DataSet/No-Mask"

mask_images = [load_and_preprocess_image(os.path.join(mask_image_dataset_url, img)) for img in os.listdir(mask_image_dataset_url)]
no_mask_images = [load_and_preprocess_image(os.path.join(no_mask_image_dataset_url, img)) for img in os.listdir(no_mask_image_dataset_url)]

# Create labels: 1 for mask, 0 for no mask
mask_labels = [1] * len(mask_images)
no_mask_labels = [0] * len(no_mask_images)

# Combine datasets
all_images = np.array(mask_images + no_mask_images)
all_labels = np.array(mask_labels + no_mask_labels)

# Convert labels to one-hot encoding
all_labels = tf.keras.utils.to_categorical(all_labels, num_classes=2)

# Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(50, (3, 3), activation='relu', input_shape=(50, 50, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.2),
              loss='mean_squared_error')

# Train the model
model.fit(all_images, all_labels, epochs=100)

# Save the model
model.save('mask_detection_model.h5')
print("Model saved as 'mask_detection_model.h5'")

# Testing with a new image
def predict_image(image_path, model):
    image = load_and_preprocess_image(image_path)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    prediction = model.predict(image)
    print(f"Prediction: {prediction}")

# Load and test with a mask image
mask_test_image = os.path.join(mask_image_dataset_url, os.listdir(mask_image_dataset_url)[0])
predict_image(mask_test_image, model)

# Load and test with a no-mask image
no_mask_test_image = os.path.join(no_mask_image_dataset_url, os.listdir(no_mask_image_dataset_url)[0])
predict_image(no_mask_test_image, model)
