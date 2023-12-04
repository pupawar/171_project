from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
import numpy as np
import os
import io
from PIL import Image, ImageDraw
import threading
import webbrowser

app = Flask(__name__)

UPLOAD_FOLDER = 'Website/static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = load_model('Website/NewResNet50.h5')

@app.route('/')
def index():
    return render_template('index.html')

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

    if len(image.shape) == 2 or image.shape[2] == 1:  # if grayscale
        pil_image = Image.fromarray(np.uint8(image)).convert('RGB')
        print('gray')
    else:
        pil_image = Image.fromarray(np.uint8(image))
        print('color')

    draw = ImageDraw.Draw(pil_image)

    keypoints = keypoints.reshape(-1, 2)
    radius = 1
    for point in keypoints:
        x, y = point
        # Define the bounding box for the circle
        # The bounding box is a square with sides of length radius*2
        upper_left = (x - radius, y - radius)
        bottom_right = (x + radius, y + radius)
        draw.ellipse([upper_left, bottom_right], fill='red', outline='red')


    # Save the resized image
    result_path = os.path.join(UPLOAD_FOLDER, 'prediction.png')
    pil_image.save(result_path)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return 'No image file provided.'

    image = request.files['image']
    
    if image:
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], '0.jpg')
        image.save(image_path)

        img = Image.open(image_path).convert('L')
        img = img.resize((96, 96))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=-1)
        img_array = np.expand_dims(img_array, axis=0)
        
        test_img = load_images()

        test_img_array = np.stack(test_img, axis=0)
        x_test = test_img_array.reshape(-1, 96, 96, 1).astype('float64')

        y_test = model.predict(x_test).astype('float64')
        image = np.squeeze(x_test, axis=(0, 3))

        show(image, y_test)

        return 'prediction.png'

    return 'Error during prediction.'

def open_browser():
    webbrowser.open_new('http://127.0.0.1:5000/')

if __name__ == '__main__':
    if os.environ.get("WERKZEUG_RUN_MAIN") != "true":
        threading.Timer(1.25, open_browser).start()
    app.run(debug=True)