
import os
import os.path
import errno

def map_files(files):
	output = {}
	for f in files:
		basename = os.path.splitext(os.path.basename(f))[0]
		output[basename] = f
	return output


def mkpath(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: 
        	raise
