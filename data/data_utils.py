import spacy
import json
import os

import torch
from torch.utils.data import DataLoader

from utils import validate_path
from data.dataset import NGramDataset

nlp = spacy.load('en')

SOS_TOKEN = '<SOS>'
EOS_TOKEN = '<EOS>'
PAD_TOKEN = '<PAD>'

DATA_DIR = 'data/pg17.txt'
token_set_name = 'idx_to_token.json'


def contains_numeric_char(token):
	for c in token:
		if c.isnumeric():
			return True
	return False


def tokenize(text):
	tokens = [token.text for token in nlp.tokenizer(text) if not contains_numeric_char(token.text)]
	# TODO: make everything lowercase?
	# TODO: organize by sentence?
	return tokens


def load_data(context_window, batch_size, device, testing=False, n_batches=None, data_dir=DATA_DIR, verbose=True):
	with open(data_dir, 'r') as fp:
		data = fp.read()
	tokens = tokenize(data)
	vocab = set(tokens)

	idx_to_token, token_to_idx = make_token_set(vocab)
	tokens = [token_to_idx[token] for token in tokens]

	dataset, generator = make_data(tokens, context_window, batch_size, device, testing, n_batches)

	if verbose:
		print('Loading Data:')
		print(f'\t{len(dataset)} instances, {len(vocab)} vocab size, {len(generator)} batches\n')

	return tokens, idx_to_token, dataset, generator


def make_data(data, context_window, batch_size, device, testing=False, n_batches=None):
	dataset = NGramDataset(
		data,
		context_window=context_window,
		device=device,
		size=batch_size * n_batches if testing else None
	)
	generator = DataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=True,
		drop_last=True)
	return dataset, generator


def make_token_set(vocab):
	idx_to_token = {}
	token_to_idx = {}
	for i, token in enumerate(vocab):
		idx_to_token[i] = token
		token_to_idx[token] = i
	return idx_to_token, token_to_idx


def load_token_set(token_dir):
	path = os.path.join(token_dir, token_set_name)
	if not os.path.exists(path):
		raise FileNotFoundError(f'Token set file ({token_set_name}) was not found in the directory ({token_dir})')

	with open(path, 'r') as fp:
		idx_to_token_raw = json.load(fp)

	idx_to_token = {}
	token_to_idx = {}
	for idx, token in idx_to_token_raw.items():
		idx_to_token[int(idx)] = token
		token_to_idx[token] = int(idx)

	return idx_to_token, token_to_idx


def save_token_set(idx_to_token, token_dir, pretty=True):
	path = os.path.join(token_dir, token_set_name)
	validate_path(path, is_dir=False)
	with open(path, 'w') as fp:
		json.dump(idx_to_token, fp, indent=2 if pretty else None)
