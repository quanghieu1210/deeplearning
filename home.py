import requests
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from io import BytesIO

# Load an image from URL
url = "Link ảnh"
response = requests.get(url)
image = Image.open(BytesIO(response.content))

# Display the image
plt.imshow(image)
plt.axis("off")
plt.show()

# Preprocess the image
image = image.resize((299, 299))  # Resize the image to match the input size of InceptionV3
image = np.array(image) / 255.0  # Normalize pixel values to [0, 1]
image = tf.convert_to_tensor(image, dtype=tf.float32)  # Convert image to tensor
image = tf.expand_dims(image, axis=0)  # Add batch dimension

# Load the pre-trained InceptionV3 model
model = tf.keras.applications.InceptionV3(weights='imagenet')

# Make predictions
predictions = model.predict(image)

# Decode predictions
decoded_predictions = tf.keras.applications.inception_v3.decode_predictions(predictions, top=1)[0]

# Display the top prediction
label_id, label_name, confidence = decoded_predictions[0]
print(f"The image contains a {label_name} with confidence {confidence*100:.2f}%.")
