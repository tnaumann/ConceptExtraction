from flask import Flask
from flask import render_template

import json as simplejson

app = Flask(__name__)

@app.route('/')
def index():
	return render_template("home.html")

def run():
    app.debug = True
    app.run()

if __name__ == '__main__':
    run()
