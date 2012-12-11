import os
import os.path

from model import Model
from note import Note

def main():
	# Locate the test files
	path = 'data/test_data'
	files = [os.path.join(path, f) for f in os.listdir(path)]
	
	# Get data from files
	data = []
	for txt in files:
		data += read_txt(txt)
	
	# Load a model and make a prediction for each file
	model = Model()
	model.predict(data)

if __name__ == '__main__':
	main()