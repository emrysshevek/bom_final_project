import numpy as np

from sklearn.neighbors import NearestNeighbors


class EmbeddedCalculator:

	operands = '+-*/.^'

	def __init__(self, embedding):
		self.embedding = embedding

	def parse_token(self, token):
		if isinstance(token, list):
			return np.array(token)

		try:
			return float(token)
		except TypeError:
			pass

		if token in
		return result

	def process_token(self, token):
		token = self.parse_token(token)
		if isinstance(token, float):
			return token
		else:
			return self.embedding.token_to_embedding(token)

	def process_tokens(self, *tokens):
		processed = []
		for token in tokens:
			processed.append(self.process_token(token))
		return tokens

	def evaluate(self, string):
		tokens = self.process_tokens()
		total = self.process_token()
		# tokens = self.tokenizer(string)
		# token = tokens.pop(0)
		# result = self._embed_token(tokens[0])
		# for operation, token in zip(tokens[1:-1], tokens[2:]):
		# 	token = self._embed_token(token)
		# 	if operation == '+':
		# 		result += token
		# 	elif operation == '-':
		# 		result -= token
		# 	elif operation == '*':
		# 		result *= token
		# 	elif operation == '/':
		# 		result /= token

		# result = self.embedding.query(result.reshape[1, -1], return_distance=False)
		# return result

	def add(self, word1, word2):
		word1, word2 = self.process_tokens(word1, word2)
		return word1 + word2

	def subtract(self, word1, word2):
		word1, word2 = self.process_tokens(word1, word2)
		return word1 - word2

	def mult(self, word1, word2):
		word1, word2 = self.process_tokens(word1, word2)
		return word1 * word2

	def divide(self, word1, word2):
		word1, word2 = self.process_tokens(word1, word2)
		return word1 / word2

	def dot(self, word1, word2):
		word1, word2 = self.process_tokens(word1, word2)
		assert type(word1) == np.array and type(word2) == np.array
		return word1.dot(word2)

	def pow(self, word1, word2):
		word1, word2 = self.process_tokens(word1, word2)
		return word1 ** word2

	# def dist(self, word1, word2, method='euclid'):
	# 	word1, word2 = self.process_tokens(word1, word2)
	# 	return word1 ** word2
