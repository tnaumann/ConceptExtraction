import os
import os.path
import nltk
import glob
import json
import libml

from sets import Set
from model import Model
from flask import Flask
from flask import render_template
from flask import request

app = Flask(__name__)

models_directory = os.path.join(os.path.dirname(__file__), "../../models")

@app.route('/')
def index():
	items = []

	models = glob.glob(os.path.join(models_directory, '*'))
	
	for model in models:
		name = os.path.basename(model)
		available_models = Set(os.listdir(model))

		if "model" not in available_models:
			continue

		for type in ["svm", "crf", "lin"]:
			if "model." + str(type) in available_models:
				properties = {
					"name": name,
					"type": type
				}
				items.append((json.dumps(properties), name + " - " + type.upper()))

	items = sorted(items, key = lambda t: t[1])
	return render_template("form.html", models = items)

@app.route('/process', methods=['POST', 'GET'])
def process():
	data = request.form['input']
	data = nltk.sent_tokenize(data)
	data = map(nltk.word_tokenize, data)

	properties = request.form['model']
	properties = json.loads(properties)

	model = Model.load(os.path.join(models_directory, properties["name"], "model"))
	labels = model.predict(data)
	output = None


	if properties["type"] == "svm":
		output = labels[libml.SVM]
	elif properties["type"] == "crf":
		output = labels[libml.CRF]
	elif properties["type"] == "lin":
		output = labels[libml.LIN]

	output = sum(output, [])
	data = sum(data, [])
	output = zip(data, output)

	return render_template("result.html", input = request.form["input"], model = properties["name"] + " - " + properties["type"].upper(), output = output)

def run():
    app.debug = True
    app.run(host = "0.0.0.0")

if __name__ == '__main__':
    run()
