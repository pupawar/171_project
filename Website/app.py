from flask import Flask, render_template, request
import os

app = Flask(__name__)

# Set the path for the uploaded images
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Check if the 'image' file is part of the request
    if 'image' not in request.files:
        return 'No image file provided.'

    image = request.files['image']

    # Save the uploaded image
    if image:
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
        image.save(image_path)

        # Implement your image prediction logic here
        # For demonstration purposes, let's return the file name as the prediction result
        return image.filename

    return 'Error during prediction.'

if __name__ == '__main__':
    app.run(debug=True)