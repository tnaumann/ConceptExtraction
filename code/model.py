from __future__ import with_statement

import time
import os
import pickle
import re
import subprocess
import sys
import nltk
import nltk.corpus.reader
import nltk.stem
import helper

from sets import Set
from sets import ImmutableSet

from wordshape import *

import libml

class Model:
	sentence_features = ImmutableSet(["pos", "stem_wordnet", "test_result", "prev", "next"])
	word_features = ImmutableSet(["word", "length", "mitre", "stem_porter", "stem_lancaster", "word_shape"])
	# THESE ARE FEATURES I TRIED THAT DON'T LOOK THAT PROMISING
	# I have some faith in "metric_unit" and "has_problem_form"
	# "radial_loc" may be too rare and "def_class" could be over fitting
	# "metric_unit", "radial_loc", "has_problem_form", "def_class"
	
	labels = {
		"none":0,
		"treatment":1,
		"problem":2,
		"test":3
	}
	reverse_labels = {v:k for k, v in labels.items()}
	
	@staticmethod
	def load(filename='awesome.model'):
		with open(filename) as model:
			model = pickle.load(model)
		model.filename = filename
		return model

	def __init__(self, filename='awesome.model', type=libml.ALL):
		model_directory = os.path.dirname(filename)

		if model_directory != "":
			helper.mkpath(model_directory)

		self.filename = os.path.realpath(filename)
		self.type = type
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

		label_lu = lambda l: Model.labels[l]
		labels = [map(label_lu, x) for x in labels]
		
		feat_lu = lambda f: {self.vocab[item]:f[item] for item in f}
		rows = [map(feat_lu, x) for x in rows]
		
		libml.write_features(self.filename, rows, labels, self.type)

		with open(self.filename, "w") as model:
			pickle.dump(self, model)

		libml.train(self.filename, self.type)

		
	def predict(self, data):
		rows = []
		for sentence in data:
			rows.append(self.features_for_sentence(sentence))

		feat_lu = lambda f: {self.vocab[item]:f[item] for item in f if item in self.vocab}
		rows = [map(feat_lu, x) for x in rows]
		libml.write_features(self.filename, rows, None, self.type);

		libml.predict(self.filename, self.type)
		
		labels_list = libml.read_labels(self.filename, self.type)
		
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

		tags = None
		for feature in Model.sentence_features:
			if feature not in self.enabled_features:
				continue

			if feature == "pos":
				tags = tags or nltk.pos_tag(sentence)
				for i, features in enumerate(features_list):
					tag = tags[i][1]
					features[(feature, tag)] = 1
					
			if feature == "stem_wordnet":
				tags = tags or nltk.pos_tag(sentence)
				morphy_tags = {
					'NN':nltk.corpus.reader.wordnet.NOUN,
					'JJ':nltk.corpus.reader.wordnet.ADJ,
					'VB':nltk.corpus.reader.wordnet.VERB,
					'RB':nltk.corpus.reader.wordnet.ADV}
				morphy_tags = [(w, morphy_tags.setdefault(t[:2], nltk.corpus.reader.wordnet.NOUN)) for w,t in tags]
				st = nltk.stem.WordNetLemmatizer()
				for i, features in enumerate(features_list):
					tag = morphy_tags[i]
					features[(feature, st.lemmatize(*tag))] = 1
					
			if feature == "test_result":
				for index, features in enumerate(features_list):
					right = " ".join([w for w in sentence[index:]])
					if self.is_test_result(right):
						features[(feature, None)] = 1

					
		ngram_features = [{} for i in range(len(features_list))]
		if "prev" in self.enabled_features:
			prev = lambda f: {("prev_"+k[0], k[1]): v for k,v in f.items()}
			prev_list = map(prev, features_list)
			for i in range(len(features_list)):
				if i == 0:
					ngram_features[i][("prev", "*")] = 1
				else:
					ngram_features[i].update(prev_list[i-1])
				
		if "next" in self.enabled_features:
			next = lambda f: {("next_"+k[0], k[1]): v for k,v in f.items()}
			next_list = map(next, features_list)
			for i in range(len(features_list)):
				if i == len(features_list) - 1:
					ngram_features[i][("next", "*")] = 1
				else:
					ngram_features[i].update(next_list[i+1])
		
		merged = lambda d1, d2: dict(d1.items() + d2.items())
		features_list = [merged(features_list[i], ngram_features[i]) 
			for i in range(len(features_list))]
		
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
						
			if feature == "stem_porter":
				st = nltk.stem.PorterStemmer()
				features[(feature, st.stem(word))] = 1
				
			if feature == "stem_lancaster":
				st = nltk.stem.LancasterStemmer()
				features[(feature, st.stem(word))] = 1
				
			if feature == "stem_snowball":
				st = nltk.stem.SnowballStemmer("english")
				#features[(feature, st.stem(word))] = 1
                
			if feature == "word_shape":
			    wordShapes = getWordShapes(word)
			    for i, shape in enumerate(wordShapes):
			        features[(feature + str(i), shape)] = 1
					
			if feature == "metric_unit":
				unit = 0
				if self.is_weight(word):
					unit = 1
				elif self.is_size(word):
					unit = 2
				features[(feature, None)] = unit
			
			# look for prognosis locaiton
			#if feature == "radial_loc":
			# THIS MIGHT BE BUGGED
			#	if self.is_prognosis_location(word):
			#		features[(feature, None)] = 1 
			
			if feature == "has_problem_form":
				if self.has_problem_form(word):
					features[(feature, None)] = 1
			
			if feature == "def_class":
				features[(feature, None)] = self.get_def_class(word)

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
		regex = r"^[A-Za-z]+( )*(-|--|:|was|of|\*|>|<|more than|less than)( )*[0-9]+(%)*"
		if not re.search(regex, context):
			return re.search(r"^[A-Za-z]+ was (positive|negative)", context)
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
		 regex = r".*(ic|is)$"
		 return re.search(regex, word)
	
	# checks for a definitive classification at the word level
	def get_def_class (self, word):
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
		if word.lower() in test_terms:
			return 1
		elif word.lower() in problem_terms:
			return 2
		elif word.lower() in treatment_terms:
			return 3
		return 0
	