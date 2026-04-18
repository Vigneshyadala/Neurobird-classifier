import os
from dotenv import load_dotenv
load_dotenv()

from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/')
def index():
    return '<h1>NeuroBird AI Bird Classifier</h1><p>Developed by Vignesh Yadala</p><p>App is live!</p>'

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'app': 'NeuroBird'})

if __name__ == '__main__':
    app.run(debug=True)