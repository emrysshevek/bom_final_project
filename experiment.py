import argparse
import os
import yaml
import json

import torch
import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

from data.data_utils import *
from models.model_utils import *
from utils import *


def main(model_dir):
	config_path = os.path.join(model_dir, 'config.yaml')
	with open(config_path, 'r') as fp:
		config = yaml.load(fp)
	token_set_path = os.path.join(model_dir, token_set_name)
	with open(token_set_path, 'r') as fp:
		idx_to_token_raw = json.load(fp)

	idx_to_token = {}
	token_to_idx = {}
	for idx, token in idx_to_token_raw.items():
		idx_to_token[int(idx)] = token
		token_to_idx[token] = int(idx)

	model = make_word2vec_model(
		vocab_size=len(idx_to_token),
		embed_dim=config['embed_dim'],
		n_output=20,
		layers=config['n_layers'],
		device=torch.device('cpu'),
		weights=config['load_dir'],
		verbose=True)
	embedding = model.embedding.weight.data.numpy()
	dummy_y = np.random.choice([0,1], size=embedding.shape[0])
	embedding_graph = KNeighborsClassifier(n_neighbors=10).fit(embedding, dummy_y)
	token = tokenize('God')[0]
	print(token)
	token = token_to_idx[token]
	print(token)
	neighbors = embedding_graph.kneighbors(embedding[token].reshape(1, -1), n_neighbors=10, return_distance=False)[0]
	print(neighbors)
	neighbor_tokens = [idx_to_token[neighbor] for neighbor in neighbors]
	print(neighbor_tokens)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--dir', default='results/word2vec_testing', type=str)
	args = parser.parse_args()
	main(args.dir)
