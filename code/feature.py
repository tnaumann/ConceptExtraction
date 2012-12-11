
from sets import Set
from sets import ImmutableSet

sentence_features = ImmutableSet(["pos"])
token_features = ImmutableSet(["word", "length"])
enabled_features = sentence_features | token_features

def enable_features(f):
	global enabled_features
	enabled_features = enabled_features | Set(f)

def disable_features(f):
	global enabled_features
	enabled_features = enabled_features - Set(f)

def enable_all_features():
	global enabled_features
	enabled_features = sentence_features | token_features

def disable_all_features():
	global enabled_features
	enabled_features = Set([])

def features_for_sentence(sentence):
	features_list = []

	for word in sentence:
		features_list.append(features_for_word(word))

	for feature in sentence_features:
		if feature not in enabled_features:
			continue

		if feature == "pos":
			pass
	
	return features_list

def features_for_word(word):
	features = {}

	for feature in token_features:
		if feature not in enabled_features:
			continue

		if feature == "word":
			features[(feature, word)] = 1

		if feature == "length":
			features[(feature, None)] = len(word)

	return features
