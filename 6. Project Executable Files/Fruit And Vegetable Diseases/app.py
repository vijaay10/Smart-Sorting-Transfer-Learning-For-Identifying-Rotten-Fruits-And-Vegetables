from flask import Flask, render_template, request, jsonify, url_for, redirect
from tensorflow import keras
from keras._tf_keras.keras.preprocessing.image import load_img, img_to_array

from PIL import Image
import numpy as np  
import os
import tensorflow as tf

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('healthy_vs_rotten.h5')

# Route for home/index page
@app.route('/')
def index():
    return render_template("index.html")

# Route to handle prediction
@app.route('/predict', methods=['GET', 'POST'])
def output():
    if request.method == 'POST':
        file = request.files['file']
        file_path = os.path.join("static/uploads", file.filename)
        file.save(file_path)

        # Preprocess the image
        img = load_img(file_path, target_size=(224, 224))
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = img / 255.0  # Normalize

        # Prediction
        preds = model.predict(img)
        pred_index = np.argmax(preds)

        # Class labels (must match training labels)
        labels = [
            'Apple_healthy (0)', 'Apple_Rotten (1)', 'Banana_Healthy (2)', 'Banana_Rotten (3)',
            'Bellpepper__Healthy (4)', 'Bellpepper__Rotten (5)', 'Carrot_Healthy (6)', 'Carrot_Rotten (7)',
            'Cucumber_Healthy (8)', 'Cucumber_Rotten (9)', 'Grape__Healthy (10)', 'Grape__Rotten (11)',
            'Guava_Healthy (12)', 'Guava_Rotten (13)', 'Jujube_Healthy (14)', 'Jujube_Rotten (15)',
            'Mango_Healthy (16)', 'Mango_Rotten (17)', 'Orange_Healthy (18)', 'Orange_Rotten (19)',
            'Pomegranate_Healthy (20)', 'Pomegranate_Rotten (21)', 'Potato_Healthy (22)', 'Potato_Rotten (23)',
            'Strawberry_Healthy (24)', 'Strawberry_Rotten (25)', 'Tomato__Healthy (26)', 'Tomato__Rotten (27)'
        ]

        prediction = labels[pred_index]
        print("Prediction:", prediction)

        # Return result to template
        return render_template("portfolio-details.html", prediction=prediction, image_path=file_path)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, port=2222)