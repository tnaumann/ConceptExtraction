from flask import Flask
from flask import render_template
from flask import request

import json as simplejson

app = Flask(__name__)

@app.route('/')
def index():
	return render_template("form.html")

@app.route('/process', methods=['POST', 'GET'])
def function():
	return render_template("result.html", input = request.form['input'])

def run():
    app.debug = True
    app.run()

if __name__ == '__main__':
    run()
