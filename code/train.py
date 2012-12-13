import os
import os.path
import sys
import glob
import argparse

from sets import Set
from model import Model
from note import *

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("-txt", default = '../data/concept_assertion_relation_training_data/beth/txt/*')
	parser.add_argument("-con", default = '../data/concept_assertion_relation_training_data/beth/concept/*')
	args = parser.parse_args()

	training_list = []
	txt_files = glob.glob(args.txt)
	con_files = glob.glob(args.con)

	txt_files_map = {}
	for txt_file in txt_files:
		basename = os.path.splitext(os.path.basename(txt_file))[0]
		txt_files_map[basename] = txt_file

	con_files_map = {}
	for con_file in con_files:
		basename = os.path.splitext(os.path.basename(con_file))[0]
		con_files_map[basename] = con_file

	for k in txt_files_map:
		if k in con_files_map:
			training_list.append((txt_files_map[k], con_files_map[k]))

	## Locate all the training files
	#files = []
	#for h in ['beth', 'partners']:
	#	path = os.path.join('data/concept_assertion_relation_training_data/', h)
	#	
	#	txts = os.listdir(os.path.join(path, 'txt'))
	#	cons = os.listdir(os.path.join(path, 'concept'))
	#	assert "files lined up", all(t[:-3] == c[:-3] for t, c in zip(txts, cons))
	#	
	#	txts = map(lambda f: os.path.join(path, 'txt', f), txts)
	#	cons = map(lambda f: os.path.join(path, 'concept', f), cons)
	#	files += zip(txts, cons)
	
	# Get data and labels from files
	data = []
	labels = []
	for txt, con in training_list:
		datum = read_txt(txt)
		data += datum
		labels += read_con(con, datum)
	
	# Train a model on the data and labels
	model = Model()
	model.train(data, labels)

if __name__ == '__main__':
	main()