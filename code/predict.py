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
	model.predict()

if __name__ == '__main__':
	main()