import spacy
import torch
from torch.utils.data import Dataset

nlp = spacy.load('en')

SOS_TOKEN = '<SOS>'
EOS_TOKEN = '<EOS>'
PAD_TOKEN = '<PAD>'

DATA_DIR = 'data/pg17.txt'


def load_data(data_dir=DATA_DIR):
	with open(data_dir, 'r') as fp:
		data = fp.read()
	tokens = tokenize(data)
	vocab = set(tokens)
	idx_to_token, token_to_idx = make_token_set(vocab)
	tokens = [token_to_idx[token] for token in tokens]
	return tokens, idx_to_token, token_to_idx, vocab


def make_token_set(vocab):
	idx_to_token = {}
	token_to_idx = {}
	for i, token in enumerate(vocab):
		idx_to_token[i] = token
		token_to_idx[token] = i
	return idx_to_token, token_to_idx


def tokenize(text):
	tokens = [token.text for token in nlp.tokenizer(text)]
	# TODO: possibly remove all tokens with numbers in them?
	# TODO: organize by sentence?
	return tokens
