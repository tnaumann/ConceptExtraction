
from sets import Set
from sets import ImmutableSet

sentence_features = ImmutableSet(["pos"])
token_features = ImmutableSet(["word"])
enabled_features = sentence_features | token_features

def enable_features():
	pass

def disable_features():
	pass

def enable_all_features():
	pass

def disable_all_features():
	pass

def features_for_sentence(sentence):
	features = []

	for token in sentence:
		features.append(features_for_token(token))

	# Note - some features will be index based, and others will be shared by all tokens in the sentence
	for feature in sentence_features:
		if features not in enabled_features:
			continue

		if feature == "pos":
			pass

def features_for_token(token):
	for feature in token_features:
		if feature not in enabled_features:
			continue

		if feature == "word":
			pass

class Feature:
	def __init__(self, token):
		self.token = token