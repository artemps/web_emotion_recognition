import requests
from flask import Flask, render_template, request, jsonify, abort, make_response
from utils.load_model import load_trained_model
from utils.recognition import recognize

app = Flask(__name__)
model = load_trained_model()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    try:
    	url = request.form.get('url')
    	res = requests.get(url)
    	data = recognize(model, res.content)
    	return jsonify(data)
    except Exception as e:
        print(e)
        abort(400)


@app.errorhandler(400)
def bad_request(e):
    return make_response(jsonify({'error': 'We cannot process the file sent in the request.'}), 400)


@app.errorhandler(405)
def method_not_allowed(e):
    return make_response(jsonify({'error': 'Please, use POST method for uploading images'}), 405)


if __name__ == '__main__':
    app.run()
