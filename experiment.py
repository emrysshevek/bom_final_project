import argparse
import os
import yaml
import json

import torch
import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors, KDTree

from data.data_utils import *
from models.model_utils import *
from utils import *
# from embedded_calculator import EmbeddedCalculator
from embedding import Embedding


def main(model_dir):
	np.random.seed(0)
	torch.manual_seed(0)
	e = Embedding(model_dir)
	print(e.nearest_k('god'))
	print(e.similarity('god', 'wickedly'))
	print(e.analogy('god', 'love', 'satan', k=5))
	print()


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--dir', default='results/word2vec_testing', type=str)
	args = parser.parse_args()
	main(args.dir)
