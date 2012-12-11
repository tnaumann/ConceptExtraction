import os
import os.path

from model import Model
from note import *

def main():
	# Locate all the training files
	files = []
	for h in ['beth', 'partners']:
		path = os.path.join('data/concept_assertion_relation_training_data/', h)

		txts = os.listdir(os.path.join(path, 'txt'))
		cons = os.listdir(os.path.join(path, 'concept'))
		assert "files lined up", all(t[:-3] == c[:-3] for t, c in zip(txts, cons))

		txts = map(lambda f: os.path.join(path, 'txt', f), txts)
		cons = map(lambda f: os.path.join(path, 'concept', f), cons)
		files += zip(txts, cons)
	
	# Get data and labels from files
	data = []
	for txt, con in files:
		data += (read_txt(txt), read_con(con, txt))
	data, labels = zip(*data)	#unzip
	
	# Train a model on the data and labels
	model = Model()
	model.train(data, labels)

if __name__ == '__main__':
	main()