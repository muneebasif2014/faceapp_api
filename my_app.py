from flask import Flask, request, send_file, jsonify
import cv2
import numpy as np
from keras_facenet import FaceNet
import base64
import os
import re

# ...

app = Flask(__name__)

def overlay_flag_on_image(image_data, flag_data):
    try:
        embedder = FaceNet()
        image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)

        if image is None:
            print("Image is None")
        
        # Debugging statement to check the dimensions of the image
        print("Image dimensions:", image.shape)
        
        detections = embedder.extract(image, threshold=0.95)
        flag = cv2.imdecode(np.frombuffer(flag_data, np.uint8), cv2.IMREAD_UNCHANGED)
        overlay = np.zeros_like(image)

        for detection in detections:
            box = detection['box']
            x, y, w, h = box

            # Resize the flag image to match the size of the current ROI
            resized_flag = cv2.resize(flag, (w, h))[:, :, :3]  # Remove alpha channel

            # Overlay the resized flag image on the current ROI
            overlay[y:y+h, x:x+w] = resized_flag

        alpha = 0.3
        image_new = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

        # Convert the resulting image to base64 format
        _, buffer = cv2.imencode('.jpg', image_new)
        image_base64 = base64.b64encode(buffer).decode()

        return image_base64
    except Exception as e:
        print("Error during overlay process:", str(e))
        return None


@app.route('/')
def index():
    return "render_template('index.html')"

@app.route('/overlay', methods=['POST'])
def overlay():
    try:
        # Get the base64-encoded image and flag data from the request
        image_data = base64.b64decode(pad_base64(request.form.get('image', '')))
        flag_data = base64.b64decode(pad_base64(request.form.get('flag', '')))

        # Debugging statements to check the data lengths
        print("image_data length:", len(image_data))
        print("flag_data length:", len(flag_data))

        # Perform the image overlay operation
        result_image_base64 = overlay_flag_on_image(image_data, flag_data)

        if result_image_base64 is not None:
            # Return the resulting image in base64 format
            return result_image_base64
        else:
            return jsonify({'error': 'Failed to process overlay'}), 500
    except Exception as e:
        print("Error during overlay request:", str(e))
        return jsonify({'error': 'Bad request'}), 400

def pad_base64(base64_str):
    padding = '=' * (4 - (len(base64_str) % 4))
    return base64_str + padding

# ...



if __name__ == '__main__':
    app.run(host='192.168.10.10', port=5000)
