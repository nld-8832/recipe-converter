from flask import Flask, request, render_template
from cv2 import cv2

import numpy as np
import io
import json
import requests

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/convert', methods=['POST'])
def convert_recipe():
    try:
        #Preprocess image stuff
        img = request.files['imageFile']
        img.save('input.jpg')
        img = cv2.imread("input.jpg")

        #Pass to API
        url_api = "https://api.ocr.space/parse/image"
        _, compressedImage = cv2.imencode(".jpg", img, [1, 90]) #Free tier API only allows image <1MB
        file_bytes = io.BytesIO(compressedImage)

        result = requests.post(url_api, files = {"screenshot.jpg": file_bytes}, data = {"apikey": "ff0f1158f788957", "language": "eng"})
        result = result.content.decode()
        result = json.loads(result)
        parsed_results = result.get("ParsedResults")[0]
        text_detected = parsed_results.get("ParsedText")
   
        print("Success")

    except Exception as e:
        #Store error to pass to the web page
        message = "ErrorClass: {}, Argument: {} and Traceback details are: {}".format(e.__class__,e.args,e.__doc__)
        text_detected = "None cuz error."
        print(message)

    return render_template('index.html', text_detected=text_detected, imageFile=img)

if __name__ == "__main__":
    app.run(debug=True)