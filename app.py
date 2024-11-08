import os
import numpy as np
import cv2
import tensorflow as tf
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# Load the model
def load_model():
    try:
        model = tf.keras.models.load_model('EYE.h5')
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


MODEL = load_model()


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(image_path):
    try:
        # Read the image
        img = cv2.imread(image_path)

        # Resize the image
        img_resized = cv2.resize(img, (224, 224))

        # Normalize the image
        img_normalized = img_resized / 255.0

        # Add batch dimension
        img_preprocessed = np.expand_dims(img_normalized, axis=0)

        return img_preprocessed
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None


def get_prediction(model, preprocessed_image):
    try:
        # Make prediction
        predictions = model.predict(preprocessed_image)

        # Get the predicted class
        predicted_class = np.argmax(predictions, axis=1)[0]

        # Define class labels
        class_labels = [
            "No Disease",
            "Mild Condition",
            "Moderate Condition",
            "Severe Condition"
        ]

        # Get prediction probability
        prediction_prob = predictions[0][predicted_class] * 100

        return {
            'class': predicted_class,
            'label': class_labels[predicted_class],
            'probability': round(prediction_prob, 2)
        }
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None


@app.route('/eye')
def index():
    return render_template('eye_detection.html')

@app.route('/ear')
def ear():
    return render_template('ear_detection.html')

@app.route('/skin')
def skin():
    return render_template('skin_detection.html')

@app.route('/nose')
def nose():
    return render_template('nose_detection.html')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/eye_disease_detection', methods=['POST'])
def eye_disease_detection():
    # Check if model is loaded
    if MODEL is None:
        return render_template('error.html', error="Model could not be loaded")

    # Check if file is present
    if 'image' not in request.files:
        return render_template('eye_detection.html', error="No file uploaded")

    file = request.files['image']

    # Check if filename is empty
    if file.filename == '':
        return render_template('eye_detection.html', error="No selected file")

    # Check if file is allowed
    if file and allowed_file(file.filename):
        # Save the file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            # Preprocess the image
            preprocessed_image = preprocess_image(filepath)

            if preprocessed_image is None:
                return render_template('error.html', error="Image preprocessing failed")

            # Get prediction
            prediction = get_prediction(MODEL, preprocessed_image)

            if prediction is None:
                return render_template('error.html', error="Prediction failed")

            # Render result template
            return render_template('eye_detection_result.html',
                                   predicted_class=prediction['class'],
                                   prediction=f"{prediction['label']} (Confidence: {prediction['probability']}%)")

        except Exception as e:
            return render_template('error.html', error=str(e))

    return render_template('eye_detection.html', error="Invalid file type")


@app.route('/ear_disease_detection', methods=['POST'])
def ear_disease_detection():
    # Check if model is loaded
    if MODEL is None:
        return render_template('error.html', error="Model could not be loaded")

    # Check if file is present
    if 'image' not in request.files:
        return render_template('ear_detection.html', error="No file uploaded")

    file = request.files['image']

    # Check if filename is empty
    if file.filename == '':
        return render_template('ear_detection.html', error="No selected file")

    # Check if file is allowed
    if file and allowed_file(file.filename):
        # Save the file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            # Preprocess the image
            preprocessed_image = preprocess_image(filepath)

            if preprocessed_image is None:
                return render_template('error.html', error="Image preprocessing failed")

            # Get prediction
            prediction = get_prediction(MODEL, preprocessed_image)

            if prediction is None:
                return render_template('error.html', error="Prediction failed")

            # Render result template
            return render_template('ear_detection_result.html',
                                   predicted_class=prediction['class'],
                                   prediction=f"{prediction['label']} (Confidence: {prediction['probability']}%)")

        except Exception as e:
            return render_template('error.html', error=str(e))

    return render_template('ear_detection.html', error="Invalid file type")

@app.route('/skin_disease_detection', methods=['POST'])
def skin_disease_detection():
    # Check if model is loaded
    if MODEL is None:
        return render_template('error.html', error="Model could not be loaded")

    # Check if file is present
    if 'image' not in request.files:
        return render_template('skin_detection.html', error="No file uploaded")

    file = request.files['image']

    # Check if filename is empty
    if file.filename == '':
        return render_template('skin_detection.html', error="No selected file")

    # Check if file is allowed
    if file and allowed_file(file.filename):
        # Save the file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            # Preprocess the image
            preprocessed_image = preprocess_image(filepath)

            if preprocessed_image is None:
                return render_template('error.html', error="Image preprocessing failed")

            # Get prediction
            prediction = get_prediction(MODEL, preprocessed_image)

            if prediction is None:
                return render_template('error.html', error="Prediction failed")

            # Render result template
            return render_template('skin_detection_result.html',
                                   predicted_class=prediction['class'],
                                   prediction=f"{prediction['label']} (Confidence: {prediction['probability']}%)")

        except Exception as e:
            return render_template('error.html', error=str(e))

    return render_template('skin_detection.html', error="Invalid file type")

@app.route('/nose_disease_detection', methods=['POST'])
def nose_disease_detection():
    # Check if model is loaded
    if MODEL is None:
        return render_template('error.html', error="Model could not be loaded")

    # Check if file is present
    if 'image' not in request.files:
        return render_template('nose_detection.html', error="No file uploaded")

    file = request.files['image']

    # Check if filename is empty
    if file.filename == '':
        return render_template('nose_detection.html', error="No selected file")

    # Check if file is allowed
    if file and allowed_file(file.filename):
        # Save the file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            # Preprocess the image
            preprocessed_image = preprocess_image(filepath)

            if preprocessed_image is None:
                return render_template('error.html', error="Image preprocessing failed")

            # Get prediction
            prediction = get_prediction(MODEL, preprocessed_image)

            if prediction is None:
                return render_template('error.html', error="Prediction failed")

            # Render result template
            return render_template('nose_detection_result.html',
                                   predicted_class=prediction['class'],
                                   prediction=f"{prediction['label']} (Confidence: {prediction['probability']}%)")

        except Exception as e:
            return render_template('error.html', error=str(e))

    return render_template('nose_detection.html', error="Invalid file type")


@app.errorhandler(Exception)
def handle_exception(e):
    return render_template('error.html', error=str(e)), 500


if __name__ == '__main__':
    app.run(debug=True)