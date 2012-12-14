import os
import os.path
import sys
import glob
import argparse
import helper

import libml
from model import Model
from note import *

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("-i", 
		dest = "input", 
		help = "The input files to predict", 
		default = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data/test_data/*')
	)

	parser.add_argument("-o", 
		dest = "output", 
		help = "The directory to write the output", 
		default = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data/test_predictions')
	)

	parser.add_argument("-m",
		dest = "model",
		help = "The model to use for prediction",
		default = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../model/awesome.model')
	)

	
	args = parser.parse_args()

	# Locate the test files
	files = glob.glob(args.input)

	# Load a model and make a prediction for each file
	path = args.output
	helper.mkpath(args.output)

	model = Model(filename = args.model)
	for txt in files:
		data = read_txt(txt)
		labels = model.predict(data)
		con = os.path.split(txt)[-1]
		con = con[:-3] + 'con'
		
		for t in libml.bits(libml.ALL):
			if t == libml.SVM:
				helper.mkpath(os.path.join(args.output, "svm"))
				con_path = os.path.join(path, "svm", con)
			if t == libml.LIN:
				helper.mkpath(os.path.join(args.output, "lin"))
				con_path = os.path.join(path, "lin", con)
			if t == libml.CRF:
				helper.mkpath(os.path.join(args.output, "crf"))
				con_path = os.path.join(path, "crf", con)
				
			write_con(con_path, data, labels[t])

if __name__ == '__main__':
	main()