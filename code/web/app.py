from flask import Flask
from flask import render_template
from flask import request

import json as simplejson

import os
import os.path

app = Flask(__name__)

@app.route('/')
def index():

	models_directory = os.path.join(os.path.dirname(__file__), "../../models")
	models = os.listdir(models_directory)
	return render_template("form.html", models = models)

@app.route('/process', methods=['POST', 'GET'])
def function():
	return render_template("result.html", input = request.form['input'], model = request.form['model'])

def run():
    app.debug = True
    app.run()

if __name__ == '__main__':
    run()
