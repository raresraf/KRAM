import sys
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import json

sys.path.insert(0, '../../KGQA/LSTM/')

from engine import Engine

app = Flask(__name__)

app.config['CORS_HEADERS'] = 'Content-Type'


@app.route('/answer', methods=['POST'])
@cross_origin()
def get_videos():
    engine = Engine()

    if request.method == 'POST':
        data = request.get_json()
        answer = engine.answer(data['question'])

        return jsonify({'answer': answer})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)