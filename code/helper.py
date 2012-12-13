
import os
import os.path

def map_files(files):
	output = {}
	for f in files:
		basename = os.path.splitext(os.path.basename(f))[0]
		output[basename] = f
	return output