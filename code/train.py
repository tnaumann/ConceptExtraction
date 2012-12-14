import os
import os.path
import sys
import glob
import argparse
import helper
import libml

from sets import Set
from model import Model
from note import *

def main():
	parser = argparse.ArgumentParser()

	parser.add_argument("-t", 
		dest = "txt", 
		help = "The files that contain the training examples",
		default = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data/concept_assertion_relation_training_data/beth/txt/*')
	)
	
	parser.add_argument("-c", 
		dest = "con", 
		help = "The files that contain the labels for the training examples",
		default = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data/concept_assertion_relation_training_data/beth/concept/*')
	)
	
	parser.add_argument("-m",
		dest = "model",
		help = "Path to the model that should be generated",
		default = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../model/awesome.model')
	)

	parser.add_argument("-d",
		dest = "disabled_features",
		help = "The features that should not be used",
		nargs = "+",
		default = []
	)

	parser.add_argument("--no-svm",
		dest = "no_svm",
		action = "store_true",
		help = "Disable SVM model generation",
	)

	parser.add_argument("--no-lin",
		dest = "no_lin",
		action = "store_true",
		help = "Disable LIN model generation",
	)

	parser.add_argument("--no-crf",
		dest = "no_crf",
		action = "store_true",
		help = "Disable CRF model generation",
	)

	args = parser.parse_args()

	training_list = []
	txt_files = glob.glob(args.txt)
	con_files = glob.glob(args.con)

	txt_files_map = helper.map_files(txt_files)
	con_files_map = helper.map_files(con_files)

	for k in txt_files_map:
		if k in con_files_map:
			training_list.append((txt_files_map[k], con_files_map[k]))

	type = 0
	if not args.no_svm:
		type = type | libml.SVM

	if not args.no_lin:
		type = type | libml.LIN
	
	if not args.no_crf:
		type = type | libml.CRF

	
	# Get data and labels from files
	data = []
	labels = []
	for txt, con in training_list:
		datum = read_txt(txt)
		data += datum
		labels += read_con(con, datum)
	
	# Train a model on the data and labels
	model = Model(filename = args.model, type = type)
	model.enabled_features = model.enabled_features - Set(args.disabled_features)
	model.train(data, labels)

if __name__ == '__main__':
	main()