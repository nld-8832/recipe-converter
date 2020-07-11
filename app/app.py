from flask import Flask, request, render_template
from pathlib import Path
from natsort import natsorted, ns
from cv2 import cv2

import numpy as np
import io
import json
import requests
import glob

#from tensorflow.keras.models import load_model

import sys
sys.path.append('/models/')
import sklearn_crfsuite as crf
from app.parsing import tokenize, standardize, asfloat
from app.evaluate import getlabels
from app.training import removeiob, getfeatures
from app.preprocess import get_image, resize, preprocess_mobile_image, sentences_segmentate, resize_sentences_for_model, preprocess_for_model

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/convert', methods=['POST'])
def convert_recipe():
    try:
        img = request.files['imageFile']
        img.save('temp/input.jpg')
        img = get_image("temp/input.jpg")

        #Pass to model
        '''
        #Preprocess image stuff
        #img = resize(img)

        option = request.form['inlineRadioOptions']
        if (option == 'optionPhoto'):
            preprocess_mobile_image(img)
            new_img = get_image("temp/mobile_output.jpg")
            sentences_segmentate(new_img)
        elif (option == 'optionScreenshot'):
            sentences_segmentate(img)
        
        sentences_path = Path("temp/out_sentences")
        images = list(map(str, list(sentences_path.glob("*.jpg"))))
        images = natsorted(images, alg=ns.PATH)

        resize_sentences_for_model(images)

        #TODO: LOAD THE MODEL
        #classification_model = load_model('/models/model_kerastut01.h5', compile=False)

        #TODO: traverse out_sentences folder in backwards to predict all images
        #for image in reversed(images):
            
        #classification_predict = classification_model.predict()

        text_detected = 'Success'

        '''

        #Pass to OCR
        url_api = "https://api.ocr.space/parse/image"
        _, compressedImage = cv2.imencode(".jpg", img, [1, 90]) #Free tier API only allows image <1MB
        file_bytes = io.BytesIO(compressedImage)

        result = requests.post(url_api, files = {"screenshot.jpg": file_bytes}, data = {"apikey": "ff0f1158f788957", "language": "eng"})
        result = result.content.decode()
        result = json.loads(result)
        parsed_results = result.get("ParsedResults")[0]
        text_OCR = parsed_results.get("ParsedText")

        #OCR can't recognize Unicode fraction
        text_list = text_OCR.split('\n')
        text_list_strip = [b.replace('\r', '') for b in text_list]
        text_list_strip_1 = [b.replace('Y2', '1/2') for b in text_list_strip]
        text_list_strip_2 = [b.replace('Y3', '1/3') for b in text_list_strip_1]
        text_list_strip_3 = [b.replace('Y4', '1/4') for b in text_list_strip_2]
        text_list_strip_4 = [b.replace('Y8', '1/8') for b in text_list_strip_3]
        text_list_strip = filter(None, text_list_strip_4)

        #Set labels to be compared
        INGREDIENTS = ["B-INGR", "I-INGR"]
        QTY = ["B-QTY", "I-QTY"]
        UNITS = ["B-UNIT", "I-UNIT"]
        IMPERIALS = ["tablespoon", "teaspoon", "cup", "ounce", "pound"]

        converted_grams = []

        #Pass to CRF
        crf_model = "models/model.crfsuite"
        for item in text_list_strip:
            ingredient_list = []
            qty_list = []
            unit_list = []

            tokens = tokenize(item, preprocess=True)
            preds = getlabels(item, crf_model)
            print(tokens)
            print(preds)
            #displaytags(tokens, preds)

            for token, pred in zip(tokens, preds):
                if (pred in INGREDIENTS):
                    ingredient_list.append(token)
                elif (pred in QTY):
                    qty_list.append(token)
                elif (pred in UNITS):
                    unit_list.append(token)

                ingredient_string = ' '.join(ingredient_list)
                qty_string = ' '.join(qty_list)
                qty_string = asfloat(qty_string)
                unit_string = ' '.join(unit_list)
                unit_string = standardize(unit_string)
                #print(unit_string)
            
            if (unit_string in IMPERIALS):
                data = {
                    "apiKey": "ef6e10dabdeb46d7a0088c6ea82b55ed",
                    "ingredientName": ingredient_string,
                    "sourceAmount": qty_string,
                    "sourceUnit": unit_string,
                    "targetUnit": "grams"
                }

                datajson = json.dumps(data)
                responselogin = requests.get(url='https://api.spoonacular.com/recipes/convert', params=data)
                #print(responselogin.status_code)
                result = responselogin.json()
                print(result)

                grams = result['answer']
                if(grams is not None):
                    converted_grams.append(grams)
            print('-'*100)
        
        text_detected = "\n".join(converted_grams)
        print("Success")

    except Exception as e:
        #Store error to pass to the web page
        message = "ErrorClass: {}, Argument: {} and Traceback details are: {}".format(e.__class__,e.args,e.__doc__)
        text_detected = message
        print(text_detected)

    return render_template('index.html', text_detected=text_detected)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)