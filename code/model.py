from __future__ import with_statement

import time
import os
import pickle
import re
import subprocess
import sys
import nltk
import helper

from sets import Set
from sets import ImmutableSet

import libml

class Model:
	sentence_features = ImmutableSet(["pos"])
	word_features = ImmutableSet(["word", "length", "mitre"])
	
	labels = {
		"none":0,
		"treatment":1,
		"problem":2,
		"test":3
	}
	reverse_labels = {v:k for k, v in labels.items()}

	def __init__(self, filename='awesome.model', type=libml.ALL):
		model_directory = os.path.dirname(filename)

		if model_directory != "":
			helper.mkpath(model_directory)

		self.filename = filename
		self.vocab = {}

		self.enabled_features = Model.sentence_features | Model.word_features
	
	def train(self, data, labels, type=libml.ALL):
		rows = []
		for sentence in data:
			rows.append(self.features_for_sentence(sentence))

		for row in rows:
			for features in row:
				for feature in features:
					if feature not in self.vocab:
						self.vocab[feature] = len(self.vocab) + 1

		label_lu = lambda l: Model.labels[l]
		labels = [map(label_lu, x) for x in labels]
		
		feat_lu = lambda f: {self.vocab[item]:f[item] for item in f}
		rows = [map(feat_lu, x) for x in rows]
		
		libml.write_features(self.filename, rows, labels, type)

		with open(self.filename, "w") as model:
			pickle.dump(self, model)

		libml.train(self.filename, type)

		
	def predict(self, data, type=libml.ALL):
		with open(self.filename) as model:
			self = pickle.load(model)
		
		rows = []
		for sentence in data:
			rows.append(self.features_for_sentence(sentence))

		feat_lu = lambda f: {self.vocab[item]:f[item] for item in f if item in self.vocab}
		rows = [map(feat_lu, x) for x in rows]
		libml.write_features(self.filename, rows, None, type);

		libml.predict(self.filename, type)
		
		labels_list = libml.read_labels(self.filename, type)
		
		for t, labels in labels_list.items():
			tmp = []
			for sentence in data:
				tmp.append([labels.pop(0) for i in range(len(sentence))])
				tmp[-1] = map(lambda l: l.strip(), tmp[-1])
				tmp[-1] = map(lambda l: Model.reverse_labels[int(l)], tmp[-1])
			labels_list[t] = tmp

		return labels_list

	def features_for_sentence(self, sentence):
		features_list = []

		for word in sentence:
			features_list.append(self.features_for_word(word))

		for feature in Model.sentence_features:
			if feature not in self.enabled_features:
				continue

			if feature == "pos":
				tags = nltk.pos_tag(sentence)
				for index, features in enumerate(features_list):
					tag = tags[index][1]
					features[("pos", tag)] = 1

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
			
			if feature == "mitre":
				for f in Model.mitre_features:
					if re.search(Model.mitre_features[f], word):
						features[(feature, f)] = 1

		return features

	mitre_features = {
		"INITCAP" : r"^[A-Z].*$",
		"ALLCAPS" : r"^[A-Z]+$",
		"CAPSMIX" : r"^[A-Za-z]+$",
		"HASDIGIT" : r"^.*[0-9].*$",
		"SINGLEDIGIT" : r"^[0-9]$",
		"DOUBLEDIGIT" : r"^[0-9][0-9]$",
		"FOURDIGITS" : r"^[0-9][0-9][0-9][0-9]$",
		"NATURALNUM" : r"^[0-9]+$",
		"REALNUM" : r"^[0-9]+.[0-9]+$",
		"ALPHANUM" : r"^[0-9A-Za-z]+$",
		"HASDASH" : r"^.*-.*$",
		"PUNCTUATION" : r"^[^A-Za-z0-9]+$",
		"PHONE1" : r"^[0-9][0-9][0-9]-[0-9][0-9][0-9][0-9]$",
		"PHONE2" : r"^[0-9][0-9][0-9]-[0-9][0-9][0-9]-[0-9][0-9][0-9][0-9]$",
		"FIVEDIGIT" : r"^[0-9][0-9][0-9][0-9][0-9]",
		"NOVOWELS" : r"^[^AaEeIiOoUu]+$",
		"HASDASHNUMALPHA" : r"^.*[A-z].*-.*[0-9].*$ | *.[0-9].*-.*[0-9].*$",
		"DATESEPERATOR" : r"^[-/]$",
	}
    
    def is_test_result (self, context):
        # note: make spaces optional? 
        regex = r"^[A-Za-z]+( )*(-|--|—|:|was|of|\*|>|<|more than|less than)( )*[0-9]+(%)*$"
        if not re.search(regex, context):
            return r"^[A-Za-z]+ was (positive|negative)$"
        return True

    def is_weight (self, word):
        regex = r"^[0-9]*(mg|g|milligrams|grams)$"
        return re.search(regex, word)
        
    def is_size (self, word): 
        regex = r"^[0-9]*(mm|cm|millimeters|centimeters)$"
        return re.search(regex, word)

    def is_prognosis_location (self, word):
        regex = r"^(c|C)[0-9]+(-(c|C)[0-9]+)*$"
        return re.search(regex, word)
    
    def has_problem_form (self, word):
         regex = r"^[A-Za-z]+(ic|is)$"
         return re.search(regex, word)
        
    test_terms = {
        "eval", "evaluation", "evaluations",
        "sat", "sats", "saturation", 
        "exam", "exams", 
        "rate", "rates",
        "test", "tests", 
        "xray", "xrays", 
        "screen", "screens", 
        "level", "levels",
        "tox"
    }
    
    problem_terms = {
        "swelling", 
        "wound", "wounds", 
        "symptom", "symptoms", 
        "shifts", "failure", 
        "insufficiency", "insufficiencies",
        "mass", "masses", 
        "aneurysm", "aneurysms",
        "ulcer", "ulcers",
        "trama", "cancer",
        "disease", "diseased",
        "bacterial", "viral",
        "syndrome", "syndromes",
        "pain", "pains"
        "burns", "burned",
        "broken", "fractured"
    }
    
    treatment_terms = {
        "therapy", 
        "replacement",
        "anesthesia",
        "supplement", "supplemental",
        "vaccine", "vaccines"
        "dose", "doses",
        "shot", "shots",
        "medication", "medicine",
        "treament", "treatments"
    }
	