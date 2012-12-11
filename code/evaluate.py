import os
import os.path

from collections import defaultdict

def main():
	# Locate the files
	path = 'data/test_data'
	txts = [f for f in os.listdir(path) if f[-3:] == 'txt']
	cons = [f for f in os.listdir(path) if f[-3:] == 'con']
	assert "files lined up", all(t[:-3] == c[:-3] for t, c in zip(txts, cons))
	txts = map(lambda f: os.path.join(path, f), txts)
	cons = map(lambda f: os.path.join(path, f), cons)
	
	path = 'data/reference_standard_for_test_data'
	refs = [os.path.join(path, f) for f in os.listdir(path)]
	assert "refs lined up", all(r.split(os.sep)[-1] == c.split(os.sep)[-1] for r, c in zip(refs, cons))
	
	files = zip(txts, cons, gold)
	
	# Do the comparison
	confusion = defaultdict(lambda : defaultdict(int))
	for txt, con, ref in files:
		txt = read_txt(txt)
		for c, r in zip(read_con(con, txt), read_con(ref, txt)):
			confusion[r][c] += 1
			
	print confusion
		
	
if __name__ == '__main__':
	main()