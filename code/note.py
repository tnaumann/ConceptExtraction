from __future__ import with_statement

class Note:
	def __init__(self, txt, con=None):
		self.sents = []
		with open(txt) as f:
			for line in f:
				self.sents.append([[w, "none"] for w in line.split()])
				
		if con:
			with open(con) as f:
				for line in f:
					c, t = line.split('||')
					t = t[3:-2]
					c = c.split()
					start = c[-2].split(':')
					end = c[-1].split(':')
					assert "concept spans one line", start[0] == end[0]
					l = int(start[0]) - 1
					start = int(start[1])
					end = int(end[1])
					
					for i in range(start, end + 1):
						self.sents[l][i][1] = t

	def __iter__(self):
		return iter(self.sents)
		