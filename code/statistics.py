import os
import os.path
import sys
import argparse
import glob
import helper

from note import *

def main():
	data_paths = {
		'test': ('../data/test_data/*', '../data/reference_standard_for_test_data/concepts/'),
		'train': ('../data/concept_assertion_relation_training_data/merged/txt/*', '../data/concept_assertion_relation_training_data/merged/concept')
	}
	
	for type, paths in data_paths.items():
		full_path = lambda f: os.path.join(os.path.dirname(os.path.realpath(__file__)), f)
		args_txt = full_path(paths[0])
		args_ref = full_path(paths[1])
	
		txt_files = glob.glob(args_txt)
		ref_files = os.listdir(args_ref)
		ref_files = map(lambda f: os.path.join(args_ref, f), ref_files)

		txt_files_map = helper.map_files(txt_files)
		ref_files_map = helper.map_files(ref_files)
		
		files = []
		for k in txt_files_map:
			if k in ref_files_map:
				files.append((txt_files_map[k], ref_files_map[k]))
		
		labels = {}
		for txt, ref in files:
			txt = read_txt(txt)
			for r in read_con(ref, txt):
				for r in r:
					if r not in labels:
						labels[r] = 0
					labels[r] += 1
					
		print type, labels
	
if __name__ == "__main__":
	main()
				