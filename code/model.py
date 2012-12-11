from __future__ import with_statement

import pickle
from sets import Set
from sets import ImmutableSet

class Model:
	class Type:
		BOTH = 0
		SVM = 1
		CRF = 2

	sentence_features = ImmutableSet(["pos"])
	word_features = ImmutableSet(["word", "length"])
	
	labels = {
		"none":0,
		"treatment":1,
		"problem":2,
		"test":3
	}

	def __init__(self, filename='awesome.model', type=Type.BOTH):
		self.filename = filename
		self.vocab = {}

		self.enabled_features = Model.sentence_features | Model.word_features
	
	def train(self, data, labels):
		rows = []
		for sentence in data:
			rows.append(self.features_for_sentence(sentence))

		for row in rows:
			for features in row:
				for feature in features:
					if feature not in self.vocab:
						self.vocab[feature] = len(self.vocab) + 1

		with open(self.filename + ".svm", "w") as svm:
			with open(self.filename + ".crf", "w") as crf:

				for sentence_index in range(len(rows)):
					sentence = rows[sentence_index]
					sentence_labels = labels[sentence_index]

					if len(sentence) != len(sentence_labels):
						raise "Dimension mismatch"

					for word_index in range(len(sentence)):
						features = sentence[word_index]
						label = sentence_labels[word_index]

						columns = {}
						svm_line = [str(Model.labels[label])]
						crf_line = [str(Model.labels[label])]

						for item in features:
							if item in self.vocab:
								columns[self.vocab[item]] = features[item]

						for key in sorted(columns.iterkeys()):
							svm_line.append(str(key) + ":" + str(columns[key]))
							crf_line.append(str(key) + "=" + str(columns[key]))

						svm.write(" ".join(svm_line))
						crf.write(" ".join(crf_line))

						svm.write("\n")
						crf.write("\n")

					crf.write("\n")

		with open(self.filename, "a") as model:
			pickle.dump(self, model)
		
	def predict(self, data):
		pass

	def features_for_sentence(self, sentence):
		features_list = []

		for word in sentence:
			features_list.append(self.features_for_word(word))

		for feature in Model.sentence_features:
			if feature not in self.enabled_features:
				continue

			if feature == "pos":
				pass

		return features_list

	def features_for_word(self, word):
		features = {}

		for feature in Model.word_features:
			if feature not in self.enabled_features:
				continue

			if feature == "word":
				features[(feature, word)] = 1

			if feature == "length":
				features[(feature, None)] = len(word)

		return features

			
	