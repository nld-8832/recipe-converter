import requests
import json
from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/convert', methods=['GET'])
def convert():
    ingredientName = request.args.get('name')
    sourceAmount = request.args.get('qty')
    sourceUnit = request.args.get('unit')
    targetUnit = request.args.get('targetUnit')

    data = {
        "apiKey": "ef6e10dabdeb46d7a0088c6ea82b55ed",
        "ingredientName": ingredientName,
        "sourceAmount": sourceAmount,
        "sourceUnit": sourceUnit,
        "targetUnit": targetUnit
    }

    datajson = json.dumps(data)
    responselogin = requests.get(url='https://api.spoonacular.com/recipes/convert', params=data )
    print(responselogin.status_code)
    return responselogin.json()


if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')