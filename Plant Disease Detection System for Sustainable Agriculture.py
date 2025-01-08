from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)
model = load_model('prediction_plant.h5')  # Load your pre-trained model
class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
                    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
                    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
                    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
                    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
                    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
                    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
                    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
                    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                      'Tomato___healthy']

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def process_and_predict(image_path, add_rectangle=False):
    """Read, process, and predict the tumor type."""
    read = cv2.imread(image_path)
    resized = cv2.resize(read, (150, 150))  # Resize to model input size
    resized = np.expand_dims(resized, axis=0)  # Add batch dimension
    prediction = model.predict(resized)
    predicted_class_index = np.argmax(prediction)
    predicted_class_name = class_name[predicted_class_index]

    # Draw a green rectangle only if `add_rectangle` is True
    if add_rectangle:
        height, width, _ = read.shape
        start_point = (int(width * 0.1), int(height * 0.1))
        end_point = (int(width * 0.9), int(height * 0.9))
        color = (0, 255, 0)  # Green color
        thickness = 3
        read = cv2.rectangle(read, start_point, end_point, color, thickness)

        # Save the image with the rectangle
        result_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result.jpg')
        cv2.imwrite(result_path, read)
        return predicted_class_name, result_path

    return predicted_class_name, image_path


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' in request.files:
        file = request.files['file']
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            add_rectangle = request.form.get('from_camera') == 'true'
            predicted_class_name, result_image = process_and_predict(file_path, add_rectangle)

            return jsonify({
                "prediction": predicted_class_name,
                "result_image": result_image
            })
    return jsonify({"error": "No file uploaded"})


if __name__ == '__main__':
    app.run(debug=True)
