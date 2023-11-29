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

def sort_pattern(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def sort_pattern(filename):
    # Extract numerical part of the filename for sorting, assuming the format 'name_number.jpg'
    num_part = ''.join(filter(str.isdigit, filename))
    return int(num_part) if num_part.isdigit() else float('inf')  # Return a large number if no digits found

def load_images(path):
    images = []
    file_list = os.listdir(path)
    sorted_files = sorted(file_list, key=sort_pattern)

    for filename in sorted_files:
        if filename.endswith(".jpg"):
            img_path = os.path.join(path, filename)
            # Open the image and convert it to grayscale
            img = Image.open(img_path).convert('L')
            # Convert the grayscale image to RGB for 96 96 3 cases. If 96 96 1, then just comment this line.
            #img_rgb = img.convert('RGB')# The reason why we convert it to RGB is because
            # Append the RGB image as a NumPy array to the images list
            images.append(np.array(img))
            img.close()

    return images


def show(image, keypoints):
    fig, axes = plt.subplots(1, dpi=100)
    axes.imshow(image, cmap='gray')

    keypoints = keypoints.reshape(-1, 2)

    # Plot keypoints
    for point in keypoints:
        x, y = point
        plt.plot(x, y, 'ro', markersize=5)

    #plt.tight_layout()
    #plt.show()
    result_path = UPLOAD_FOLDER + '/prediction.png'
    plt.savefig(result_path)

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
        
        test_img = load_images(UPLOAD_FOLDER)#Here just create a folder called image.

        test_img_array = np.stack(test_img, axis=0)  # Stack the list of images
        x_test = test_img_array.reshape(-1, 96, 96, 1).astype('float64')
        y_test = model.predict(x_test).astype('float64')
        image = np.squeeze(x_test, axis=(0, 3))
        show(image, y_test)

        filename = 'prediction.png'

        

        return filename

    return 'Error during prediction.'



if __name__ == '__main__':
    app.run(debug=True)