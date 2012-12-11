import os
import os.path

from model import Model
from note import *

def main():
	# Locate the test files
	path = 'data/test_data'
	files = [os.path.join(path, f) for f in os.listdir(path)]
	
	# Load a model and make a prediction for each file
	path = 'data/test_predictions'
	model = Model()
	for txt in files:
		data = read_txt(txt)
		labels = model.predict(data)
		con = txt.split(os.sep)[-1]
		con = os.path.join(path, con[:-3] + 'con')
		write_con(con, data, labels)

if __name__ == '__main__':
	main()