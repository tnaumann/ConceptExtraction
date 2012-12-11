import os
import os.path

from model import Model

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
	
	# Compute the confusion matrix
	labels = Model.labels
	confusion = [[0] * len(labels) for e in labels]
	for txt, con, ref in files:
		txt = read_txt(txt)
		for c, r in zip(read_con(con, txt), read_con(ref, txt)):
			confusion[labels[r]][labels[c]] += 1
	
	print "Confusion Matrix"
	print "\t\t%s" % "\t".join(Model.labels.keys())
	for act, act_v in labels:
		print "%s\t%s" % (act, "\t".join([str(confusion[act_v][pre_v]) for pre, pre_v in labels]))
	print
	
	precision = []
	recall = []
	f1 = []
	print "Analysis"
	print "\t\tPrecision\tRecall\tF1"
	for lab, lab_v in labels:
		tp = confusion[lab_v][lab_v]
		fp = sum(confusion[v][lab_v] for k, v in labels)
		fn = sum(confusion[lab_v][v] for k, v in labels)
		tn = sum(confusion[v1][v2] for k1, v1 in labels 
			for k2, v2 in labels if k1 != lab and k2 != lab)
		precision += [float(tp) / (tp + fp)]
		recall += [float(tp) / (tp + fn)]
		f1 += [float(2 * tp) / (2 * tp + fp + fn)]
		print "%s\t%f\t%f" % (lab, precision[-1], recall[-1], f1[-1])
	print "--------"
	precision = sum(precision) / len(precision)
	recall = sum(recall) / len(recall)
	f1 = sum(f1) / len(f1)
	print "Average:\t%f\t%f" % (precision, recall, f1)
	
if __name__ == '__main__':
	main()