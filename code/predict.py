import os
import os.path

from model import Model
from note import Note

def main():
	# Locate the test files
	path = 'data/test_data'
	files = [os.path.join(path, f) for f in os.listdir(path)]
	
	# Load a model and make a prediction for each file
	model = Model()
	for txt in files:
		data = read_txt(txt)
		labels = model.predict(data)
		write_labels(txt[:-3] + 'con', data, labels)

if __name__ == '__main__':
	main()