from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
model = load_model("skinclassmodels.T1Dense.keras")

# Test route
@app.route('/')
def home():
    return 'Skin Model API is running!'

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    if not file:
        return jsonify({"error": "no image file"}), 400

    # Secure filename
    filename = secure_filename(file.filename)
    img_path = os.path.join('uploads', filename)

    # save image temporarily
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    file.save(img_path)

    # preprocess image for your model (Edit target_size as per your model input)
    img = image.load_img(img_path, target_size=(224, 224)) # Change `(224,224)` if your model uses different size
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x /= 255.0  # adjust based on your model training pre-processing 

    # Predict
    preds = model.predict(x)
    predicted_class = np.argmax(preds, axis=1)[0]
    class_probability = np.max(preds)

    # clean up image
    os.remove(img_path)

    # Return JSON 
    return jsonify({
        "predicted_class": int(predicted_class),
        "probability": float(class_probability)
    })

if __name__ == '__main__':
    app.run(debug=True)
