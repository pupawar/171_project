from flask import Flask, render_template, request
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
import numpy as np
import os
import io
from PIL import Image
import random
import threading
import webbrowser

app = Flask(__name__)

UPLOAD_FOLDER = r'C:\Users\wbagh\Desktop\171\171_project\Website\static\uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = load_model(r'C:\Users\wbagh\Desktop\171\171_project\Website\NewResNet50.h5')

@app.route('/')
def index():
    return render_template('index.html')

def sort_pattern(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def sort_pattern(filename):
    # Extract numerical part of the filename for sorting, assuming the format 'name_number.jpg'
    num_part = ''.join(filter(str.isdigit, filename))
    return int(num_part) if num_part.isdigit() else float('inf')  # Return a large number if no digits found

def load_images():
    images = []
    file_list = os.listdir(UPLOAD_FOLDER)
    sorted_files = sorted(file_list, key=sort_pattern)

    for filename in sorted_files:
        if filename.endswith(".jpg"):
            img_path = os.path.join(UPLOAD_FOLDER, filename)
            img = Image.open(img_path).convert('L')
            images.append(np.array(img))
            img.close()

    return images

def show(image, keypoints):
    _, axes = plt.subplots(1, dpi=100)
    axes.imshow(image, cmap='gray')

    keypoints = keypoints.reshape(-1, 2)

    for point in keypoints:
        x, y = point
        plt.plot(x, y, 'ro', markersize=5)

    axes.axis('off')

    # Save to a temporary buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0)
    plt.close()

    # Resize the image using PIL
    buffer.seek(0)
    pil_image = Image.open(buffer)
    resized_image = pil_image.resize((96, 96), Image.ANTIALIAS)

    # Save the resized image
    result_path = UPLOAD_FOLDER + '/prediction.png'
    resized_image.save(result_path)

@app.route('/predict', methods=['POST'])
def predict():
    # Check if the 'image' file is part of the request
    if 'image' not in request.files:
        return 'No image file provided.'

    image = request.files['image']
    
    # Save the uploaded image
    if image:
        # save image in uploads
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], '0.jpg')
        image.save(image_path)

        # open image and preprocess
        img = Image.open(image_path).convert('L')
        img = img.resize((96, 96))
        img_array = np.array(img)/255.0
        img_array = np.expand_dims(img_array, axis=-1)
        img_array = np.expand_dims(img_array, axis=0)
        
        # idk wtf this is for
        test_img = load_images()

        # more preprocessing :/
        test_img_array = np.stack(test_img, axis=0)
        x_test = test_img_array.reshape(-1, 96, 96, 1).astype('float64')

        # prediction
        y_test = model.predict(x_test).astype('float64')
        image = np.squeeze(x_test, axis=(0, 3))

        # display and save 'prediction.png'
        show(image, y_test)

        return 'prediction.png'

    return 'Error during prediction.'

def open_browser():
    webbrowser.open_new('http://127.0.0.1:5000/')

if __name__ == '__main__':
    if os.environ.get("WERKZEUG_RUN_MAIN") != "true":
        threading.Timer(1.25, open_browser).start()
    app.run(debug=True)