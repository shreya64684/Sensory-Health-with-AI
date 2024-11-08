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

# Load individual models for each sensory organ
def load_eye_model():
    try:
        return tf.keras.models.load_model('EYE.h5')
    except Exception as e:
        print(f"Error loading eye model: {e}")
        return None

def load_ear_model():
    try:
        return tf.keras.models.load_model('EAR.h5')
    except Exception as e:
        print(f"Error loading ear model: {e}")
        return None

def load_skin_model():
    try:
        return tf.keras.models.load_model('SKIN.h5')
    except Exception as e:
        print(f"Error loading skin model: {e}")
        return None

def load_nose_model():
    try:
        return tf.keras.models.load_model('NOSE.h5')
    except Exception as e:
        print(f"Error loading nose model: {e}")
        return None

# Load models
EYE_MODEL = load_eye_model()
EAR_MODEL = load_ear_model()
SKIN_MODEL = load_skin_model()
NOSE_MODEL = load_nose_model()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    try:
        img = cv2.imread(image_path)
        img_resized = cv2.resize(img, (224, 224))
        img_normalized = img_resized / 255.0
        img_preprocessed = np.expand_dims(img_normalized, axis=0)
        return img_preprocessed
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

# Separate prediction and processing functions for each organ
def get_eye_prediction(preprocessed_image):
    try:
        predictions = EYE_MODEL.predict(preprocessed_image)
        return process_eye_prediction(predictions)
    except Exception as e:
        print(f"Error during eye prediction: {e}")
        return None

def get_ear_prediction(preprocessed_image):
    try:
        predictions = EAR_MODEL.predict(preprocessed_image)
        return process_ear_prediction(predictions)
    except Exception as e:
        print(f"Error during ear prediction: {e}")
        return None

def get_skin_prediction(preprocessed_image):
    try:
        predictions = SKIN_MODEL.predict(preprocessed_image)
        return process_skin_prediction(predictions)
    except Exception as e:
        print(f"Error during skin prediction: {e}")
        return None

def get_nose_prediction(preprocessed_image):
    try:
        predictions = NOSE_MODEL.predict(preprocessed_image)
        return process_nose_prediction(predictions)
    except Exception as e:
        print(f"Error during nose prediction: {e}")
        return None

# Individual process prediction functions
def process_eye_prediction(predictions):
    predicted_class = np.argmax(predictions, axis=1)[0]
    class_labels = ["No Disease", "Mild Condition", "Moderate Condition", "Severe Condition"]
    prediction_prob = predictions[0][predicted_class] * 100
    return {
        'class': predicted_class,
        'label': class_labels[predicted_class],
        'probability': round(prediction_prob, 2)
    }

def process_ear_prediction(predictions):
    predicted_class = np.argmax(predictions, axis=1)[0]
    class_labels = ["No Disease", "Mild Condition", "Moderate Condition", "Severe Condition"]
    prediction_prob = predictions[0][predicted_class] * 100
    return {
        'class': predicted_class,
        'label': class_labels[predicted_class],
        'probability': round(prediction_prob, 2)
    }

def process_skin_prediction(predictions):
    predicted_class = np.argmax(predictions, axis=1)[0]
    class_labels = ["No Disease", "Mild Condition", "Moderate Condition", "Severe Condition"]
    prediction_prob = predictions[0][predicted_class] * 100
    return {
        'class': predicted_class,
        'label': class_labels[predicted_class],
        'probability': round(prediction_prob, 2)
    }

def process_nose_prediction(predictions):
    predicted_class = np.argmax(predictions, axis=1)[0]
    class_labels = ["No Disease", "Mild Condition", "Moderate Condition", "Severe Condition"]
    prediction_prob = predictions[0][predicted_class] * 100
    return {
        'class': predicted_class,
        'label': class_labels[predicted_class],
        'probability': round(prediction_prob, 2)
    }

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
    if EYE_MODEL is None:
        return render_template('error.html', error="Eye model could not be loaded")
    if 'image' not in request.files:
        return render_template('eye_detection.html', error="No file uploaded")
    file = request.files['image']
    if file.filename == '':
        return render_template('eye_detection.html', error="No selected file")
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        try:
            preprocessed_image = preprocess_image(filepath)
            if preprocessed_image is None:
                return render_template('error.html', error="Image preprocessing failed")
            prediction = get_eye_prediction(preprocessed_image)
            if prediction is None:
                return render_template('error.html', error="Prediction failed")
            return render_template('eye_detection_result.html',
                                   predicted_class=prediction['class'],
                                   prediction=f"{prediction['label']} (Confidence: {prediction['probability']}%)")
        except Exception as e:
            return render_template('error.html', error=str(e))
    return render_template('eye_detection.html', error="Invalid file type")

@app.route('/ear_disease_detection', methods=['POST'])
def ear_disease_detection():
    if EAR_MODEL is None:
        return render_template('error.html', error="Ear model could not be loaded")
    if 'image' not in request.files:
        return render_template('ear_detection.html', error="No file uploaded")
    file = request.files['image']
    if file.filename == '':
        return render_template('ear_detection.html', error="No selected file")
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        try:
            preprocessed_image = preprocess_image(filepath)
            if preprocessed_image is None:
                return render_template('error.html', error="Image preprocessing failed")
            prediction = get_ear_prediction(preprocessed_image)
            if prediction is None:
                return render_template('error.html', error="Prediction failed")
            return render_template('ear_detection_result.html',
                                   predicted_class=prediction['class'],
                                   prediction=f"{prediction['label']} (Confidence: {prediction['probability']}%)")
        except Exception as e:
            return render_template('error.html', error=str(e))
    return render_template('ear_detection.html', error="Invalid file type")

@app.route('/skin_disease_detection', methods=['POST'])
def skin_disease_detection():
    if SKIN_MODEL is None:
        return render_template('error.html', error="Skin model could not be loaded")
    if 'image' not in request.files:
        return render_template('skin_detection.html', error="No file uploaded")
    file = request.files['image']
    if file.filename == '':
        return render_template('skin_detection.html', error="No selected file")
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        try:
            preprocessed_image = preprocess_image(filepath)
            if preprocessed_image is None:
                return render_template('error.html', error="Image preprocessing failed")
            prediction = get_skin_prediction(preprocessed_image)
            if prediction is None:
                return render_template('error.html', error="Prediction failed")
            return render_template('skin_detection_result.html',
                                   predicted_class=prediction['class'],
                                   prediction=f"{prediction['label']} (Confidence: {prediction['probability']}%)")
        except Exception as e:
            return render_template('error.html', error=str(e))
    return render_template('skin_detection.html', error="Invalid file type")

@app.route('/nose_disease_detection', methods=['POST'])
def nose_disease_detection():
    if NOSE_MODEL is None:
        return render_template('error.html', error="Nose model could not be loaded")
    if 'image' not in request.files:
        return render_template('nose_detection.html', error="No file uploaded")
    file = request.files['image']
    if file.filename == '':
        return render_template('nose_detection.html', error="No selected file")
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        try:
            preprocessed_image = preprocess_image(filepath)
            if preprocessed_image is None:
                return render_template('error.html', error="Image preprocessing failed")
            prediction = get_nose_prediction(preprocessed_image)
            if prediction is None:
                return render_template('error.html', error="Prediction failed")
            return render_template('nose_detection_result.html',
                                   predicted_class=prediction['class'],
                                   prediction=f"{prediction['label']} (Confidence: {prediction['probability']}%)")
        except Exception as e:
            return render_template('error.html', error=str(e))
    return render_template('nose_detection.html', error="Invalid file type")

@app.errorhandler(Exception)
def handle_exception(e):
    return render_template('error.html', error=str(e)), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
