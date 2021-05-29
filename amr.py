class AMR:

	def __init__(self, generation):
		self.source = []
		self.target = []
		self.edges = []
		self.generation = generation
  
	def extract_edges(self):
		if self.generation:
			sources = self.source
		else:
			sources = self.target

		for src in sources:
			tokens = src.split()
			self.edges += [token.strip() for token in tokens if token.startswith(":")]
		self.edges = list(set(self.edges))
		
