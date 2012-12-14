
import itertools
import os
import os.path
import sys
import time

from subprocess import *

from model import Model

# Parameterization
n = 1
_TEST = True

###############################################################################
# Utilities
###############################################################################

# Library locations
this_path = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(this_path, "..", "models")
data_path = os.path.join(this_path, "..", "data")
our_train = os.path.join(this_path, "train.py")
our_predict = os.path.join(this_path, "predict.py")
our_evaluate = os.path.join(this_path, "evaluate.py")

# Applicable features
features = Model.sentence_features | Model.word_features
features = sorted(list(features))
print "Features:", len(features)
print "Combinations of:", n
print

# Calculate commands
def get_cmds(function):
	commands = []
	for i in range(1,n+1):
		for c in itertools.combinations(features, i):
			# commands from single feature up
			cmd_features = [f if f in c else 'X' for f in features]
			commands.append(function(cmd_features))
			
			# commands from full feature down
			cmd_features = ['X' if f in c else f for f in features]
			commands.append(function(cmd_features))

	# commands for full feature only
	commands.append(function(features))
	return commands

# Drive process execution
def execute(commands, sleep=1):
	ps = []
	for cmd in commands:
		p = Popen(cmd)
		ps.append(p)
	print "Done"
	 
	while True:
		ps_status = [p.poll() for p in ps]
		print "\tCompleted: %d/%d\r" % (len(ps_status) - ps_status.count(None), len(ps_status)),
		sys.stdout.flush()
		if all(x is not None for x in ps_status):
			break
		time.sleep(sleep)

###############################################################################
# Train
###############################################################################
def train(cmd_features):
	modelname = os.path.join(model_path, "-".join(cmd_features), "model")
	cmd = ["python", our_train, "-m", modelname]
	if _TEST:
		cmd += ["-t", os.path.join(data_path, "concept_assertion_relation_training_data", "merged", "txt", "record-105.txt")]
		cmd += ["-c", os.path.join(data_path, "concept_assertion_relation_training_data", "merged", "concept", "record-105.con")]
	return cmd
	
print "BEGIN train"
print "\tCalculating training commands...",
commands = get_cmds(train)
print "Done"
print "\tSpawning %d training commands..." % (len(commands)),
execute(commands)
print "\tCompleted training."
print "END train"
print

###############################################################################
# Predict
###############################################################################
def predict(cmd_features):
	modelname = os.path.join(model_path, "-".join(cmd_features), "model")
	cmd = ["python", our_predict, "-m", modelname]
	cmd += ["-o", os.path.join(model_path, "-".join(cmd_features), "test_predictions")]
	if _TEST:
		cmd += ["-i", os.path.join(data_path, "concept_assertion_relation_training_data", "merged", "txt", "record-105.txt")]
	else:
		cmd += ["-i", os.path.join(data_path, "test_data", "*")]
	return cmd
	
print "BEGIN predict"
print "\tCalculating prediction commands...",
commands = get_cmds(predict)
print "Done"
print "\tSpawning %d prediction commands..." % (len(commands)),
execute(commands)
print "\tCompleted prediction."
print "END predict"
print

###############################################################################
# Evaluate
###############################################################################
def evaluate(cmd_features):
	cmd = ["python", our_evaluate]
	cmd += ["-c", os.path.join(model_path, "-".join(cmd_features), "test_predictions")]
	cmd += ["-o", os.path.join(model_path, "-".join(cmd_features), "evaluation.txt")]
	if _TEST:
		cmd += ["-t", os.path.join(data_path, "concept_assertion_relation_training_data", "merged", "txt", "record-105.txt")]
		cmd += ["-r", os.path.join(data_path, "concept_assertion_relation_training_data", "merged", "concept")]
	else:
		cmd += ["-t", os.path.join(data_path, "test_data", "*")]
		cmd += ["-r", os.path.join(data_path, "reference_standard_for_test_data", "concepts")]
	return cmd
	
print "BEGIN evaluate"
print "\tCalculating evaluation commands...",
commands = get_cmds(evaluate)
print "Done"
print "\tSpawning %d evaluation commands..." % (len(commands)),
execute(commands)
print "\tCompleted evaluation."
print "END evaluate"
print