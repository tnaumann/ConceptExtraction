
import itertools
import os
import os.path
import sys
import time

from subprocess import *

from model import Model

# Library locations
this_path = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(this_path, "..", "models")
data_path = os.path.join(this_path, "..", "data")
our_train = os.path.join(this_path, "train.py")
our_predict = os.path.join(this_path, "predict.py")

n = 0
_TEST = True

features = Model.sentence_features | Model.word_features
features = sorted(list(features))
print "Features:", len(features)
print "Combinations of:", n
print
print "Calculating training commands...",
commands = []
for i in range(1,n+1):
	for c in itertools.combinations(features, i):
		# commands from single feature up
		cmd_features = [f if f in c else 'X' for f in features]
		modelname = os.path.join(model_path, "-".join(cmd_features), "model")
		cmd = ["python", our_train, "-m", modelname]
		if _TEST:
			cmd += ["-t", os.path.join(data_path, "concept_assertion_relation_training_data", "merged", "txt", "record-105.txt")]
			cmd += ["-c", os.path.join(data_path, "concept_assertion_relation_training_data", "merged", "concept", "record-105.con")]
		commands.append(cmd)
		
		# commands from full feature down
		cmd_features = ['X' if f in c else f for f in features]
		modelname = os.path.join(model_path, "-".join(cmd_features), "model")
		cmd = ["python", our_train, "-m", modelname]
		if _TEST:
			cmd += ["-t", os.path.join(data_path, "concept_assertion_relation_training_data", "merged", "txt", "record-105.txt")]
			cmd += ["-c", os.path.join(data_path, "concept_assertion_relation_training_data", "merged", "concept", "record-105.con")]
		commands.append(cmd)

# commands for full feature only
modelname = os.path.join(model_path, "-".join(features), "model")
cmd = ["python", our_train, "-m", modelname]
if _TEST:
	cmd += ["-t", os.path.join(data_path, "concept_assertion_relation_training_data", "merged", "txt", "record-105.txt")]
	cmd += ["-c", os.path.join(data_path, "concept_assertion_relation_training_data", "merged", "concept", "record-105.con")]
commands.append(cmd)

print "Done"

print "Spawning %d training commands..." % (len(commands)),
ps = []
for cmd in commands:
	p = Popen(cmd)
	ps.append(p)
print "Done"
 
while True:
	ps_status = [p.poll() for p in ps]
	print "Completed: %d/%d\r" % (len(ps_status) - ps_status.count(None), len(ps_status))
	sys.stdout.flush()
	if all(x is not None for x in ps_status):
		break
	time.sleep(1)

print "Completed training."
		




