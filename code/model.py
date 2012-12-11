from __future__ import with_statement

class Model:
	def __init__(self, filename='awesome.model'):
		self.filename = filename
		self.vocab = {}
	
	def train(self, data, labels):
		with open(self.filename, 'w') as out:
			print >>out, "Hello"
		
	def predict(self):
		pass
			
	