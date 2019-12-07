import os
import yaml
import json

import numpy as np
from scipy.spatial import distance
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import KDTree

from data.data_utils import token_set_name
from models.model_utils import make_embedding


# class Embedding:
#
# 	def __init__(self, embedding, token_to_idx, idx_to_token=None, tokenizer=None):
# 		self.embedding = embedding
# 		self.token_to_idx = token_to_idx
#
# 		if idx_to_token is not None:
# 			self.idx_to_token = idx_to_token
# 		else:
# 			self.idx_to_token = {idx: token for token, idx in token_to_idx.items()}
#
# 		if tokenizer is not None:
# 			self.tokenizer = tokenizer
# 		else:
# 			self.tokenizer = lambda x: x.split()
#
# 		self.knn = NearestNeighbors().fit(embedding)
#
# 	def token_to_embedding(self, token):
# 		return self.embedding(self.token_to_idx[token])
#
# 	def embedding_to_token(self, embedding):
# 		idx = self.knn.kneighbors(embedding.reshape(1, -1))[0]
# 		return self.idx_to_token[idx]
#
# 	def convert(self, x):
# 		if x.isinstance(str):
# 			return self.token_to_embedding(x)
# 		elif x.isinstance(list):
# 			return self.embedding_to_token(x)
# 		else:
# 			raise ValueError(f'Received unknown value type {type(x)}')


class Embedding:

	def __init__(self, model_dir):
		self.data = None
		self.embedding = None
		self.idx_to_token = {}
		self.token_to_idx = {}

		config_path = os.path.join(model_dir, 'config.yaml')
		with open(config_path, 'r') as fp:
			config = yaml.load(fp)
		token_set_path = os.path.join(model_dir, token_set_name)
		with open(token_set_path, 'r') as fp:
			idx_to_token_raw = json.load(fp)

		for idx, token in idx_to_token_raw.items():
			self.idx_to_token[int(idx)] = token
			self.token_to_idx[token] = int(idx)

		embedding = make_embedding(len(self.idx_to_token), config['embed_dim'], weights=config['load_dir'])
		self.data = embedding.weight.data.numpy()
		self.embedding = KDTree(self.data)

	def embedding_to_token(self, x):
		if len(np.array(x).shape) == 1:
			x = np.array(x).reshape(1, -1)
		idx = self.embedding.query(x, return_distance=False).item()
		return self.idx_to_token[idx]

	def token_to_embedding(self, token):
		return self.data[self.token_to_idx[token]]

	def nearest_k(self, x, k=10, return_dists=True):
		if isinstance(x, str):
			x = np.array(self.token_to_embedding(x)).reshape(1, -1)
		neighbors = self.embedding.query(x, k=k, return_distance=return_dists)

		if return_dists:
			dists, idxs = neighbors
			return [(self.idx_to_token[i], d) for d, i in zip(dists[0], idxs[0])]
		else:
			return [self.idx_to_token[i] for i in neighbors[0]]

	def similarity(self, t1, t2):
		u = self.token_to_embedding(t1).reshape(1, -1)
		v = self.token_to_embedding(t2).reshape(1, -1)

		cos_sim = cosine_similarity(u, v)[0][0]

		return (cos_sim + 1) / 2

	def analogy(self, a1, a2, b, k=1, return_dists=False):
		x1 = self.token_to_embedding(a1).reshape(1, -1)
		x2 = self.token_to_embedding(a2).reshape(1, -1)
		y1 = self.token_to_embedding(b).reshape(1, -1)
		y2 = x2 - x1 + y1
		return self.nearest_k(y2, k, return_dists)

