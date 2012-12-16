import os
import os.path
import sys
import argparse
import glob
import helper

from model import Model
from note import *

def main():
	parser = argparse.ArgumentParser()

	parser.add_argument("-t",
		help = "Test files that were used to generate predictions",
		dest = "txt",
		default = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data/test_data/*')
	)

	parser.add_argument("-c",
		help = "The directory that contains predicted concept files organized into subdirectories for svm, lin, srf",
		dest = "con",
		default = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data/test_predictions/')
	)

	parser.add_argument("-r",
		help = "The directory that contains reference gold standard concept files",
		dest = "ref",
		default = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data/reference_standard_for_test_data/concepts/')
	)
	
	parser.add_argument("-o",
		help = "Write the evaluation to a file rather than STDOUT",
		dest = "output",
		default = None
	)

	args = parser.parse_args()
	
	# output
	if args.output:
		args.output = open(args.output, "w")
	else:
		args.output = sys.stdout

	txt_files = glob.glob(args.txt)
	ref_files = os.listdir(args.ref)
	ref_files = map(lambda f: os.path.join(args.ref, f), ref_files)

	txt_files_map = helper.map_files(txt_files)
	ref_files_map = helper.map_files(ref_files)

	con_directories = os.listdir(args.con)

	for con_directory in con_directories:
		files = []
		directory_name = os.path.basename(con_directory)

		if directory_name not in ["svm", "crf", "lin"]:
			continue

		con_files = os.listdir(os.path.join(args.con, con_directory))
		con_files = map(lambda f: os.path.join(args.con, con_directory, f), con_files)
		
		con_files_map = helper.map_files(con_files)

		for k in txt_files_map:
			if k in con_files_map and k in ref_files_map:
				files.append((txt_files_map[k], con_files_map[k], ref_files_map[k]))


		# Compute the confusion matrix
		labels = Model.labels
		confusion = [[0] * len(labels) for e in labels]
		for txt, con, ref in files:
			txt = read_txt(txt)
			for c, r in zip(read_con(con, txt), read_con(ref, txt)):
				for c, r in zip(c, r):
					confusion[labels[r]][labels[c]] += 1
		


		# Display the confusion matrix
		print >>args.output, ""
		print >>args.output, ""
		print >>args.output, ""
		print >>args.output, "================"
		print >>args.output, directory_name.upper() + " RESULTS" 
		print >>args.output, "================"
		print >>args.output, ""
		print >>args.output, "Confusion Matrix"
		pad = max(len(l) for l in labels) + 6
		print >>args.output, "%s %s" % (' ' * pad, "\t".join(Model.labels.keys()))
		for act, act_v in labels.items():
			print >>args.output, "%s %s" % (act.rjust(pad), "\t".join([str(confusion[act_v][pre_v]) for pre, pre_v in labels.items()]))
		print >>args.output, ""
		
		

		# Compute the analysis stuff
		precision = []
		recall = []
		specificity = []
		f1 = []

		tp = 0
		fp = 0
		fn = 0
		tn = 0

		print >>args.output, "Analysis"
		print >>args.output, " " * pad, "Precision\tRecall\tF1"

		

		for lab, lab_v in labels.items():
			tp = confusion[lab_v][lab_v]
			fp = sum(confusion[v][lab_v] for k, v in labels.items() if v != lab_v)
			fn = sum(confusion[lab_v][v] for k, v in labels.items() if v != lab_v)
			tn = sum(confusion[v1][v2] for k1, v1 in labels.items() 
				for k2, v2 in labels.items() if v1 != lab_v and v2 != lab_v)
			precision += [float(tp) / (tp + fp + 1e-100)]
			recall += [float(tp) / (tp + fn + 1e-100)]
			specificity += [float(tn) / (tn + fp + 1e-100)]
			f1 += [float(2 * tp) / (2 * tp + fp + fn + 1e-100)]
			print >>args.output, "%s %.4f\t%.4f\t%.4f\t%.4f" % (lab.rjust(pad), precision[-1], recall[-1], specificity[-1], f1[-1])

		print >>args.output, "--------"

		precision = sum(precision) / len(precision)
		recall = sum(recall) / len(recall)
		specificity = sum(specificity) / len(specificity)
		f1 = sum(f1) / len(f1)

		print >>args.output, "Average: %.4f\t%.4f\t%.4f\t%.4f" % (precision, recall, specificity, f1)
	
if __name__ == '__main__':
	main()