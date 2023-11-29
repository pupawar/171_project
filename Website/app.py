from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import numpy as np
import pickle
import os
from PIL import Image


app = Flask(__name__)

# Set the path for the uploaded images
UPLOAD_FOLDER = r'C:\Users\parth\ECS171\171_project\Website\static\uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
model = load_model(r'C:\Users\parth\ECS171\171_project\Website\NewResNet50.h5')
print(model)
@app.route('/')
def index():
    return render_template('index.html')

def show_keypoints(image, keypoints):
    if image.shape[-1] == 1:
        image = image.squeeze()
    # Create a figure and axis
    fig, ax = plt.subplots(1)

    keypoints = keypoints.reshape(-1, 2)

    ax.imshow(image, cmap='gray')

    # Plot facial keypoints
    for point in keypoints:
        ax.plot(point[0], point[1], 'ro')
    upload_im = UPLOAD_FOLDER + '\prediction.png'
    plt.savefig(upload_im)

@app.route('/predict', methods=['POST'])
def predict():
    # Check if the 'image' file is part of the request
    if 'image' not in request.files:
        return 'No image file provided.'

    image = request.files['image']
    
    # Save the uploaded image
    if image:
        filename = secure_filename(image.filename)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(image_path)
        img = Image.open(image_path).convert('L')
        img = img.resize((96, 96 ))  # Resize to match the model's expected sizing
        img_array = np.array(img)/255.0
        img_array = np.expand_dims(img_array, axis=-1)
        img_array = np.expand_dims(img_array, axis=0)
        

        # Make predictions
        predictions = model.predict(img_array).flatten()
        show_keypoints(img_array, predictions)

        filename = 'prediction.png'

        

        return filename

    return 'Error during prediction.'



if __name__ == '__main__':
    app.run(debug=True)