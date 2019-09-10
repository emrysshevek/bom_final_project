import os
from copy import deepcopy

import torch

from models.word2vec import Word2Vec
from utils import validate_path

model_name = 'model'
embedding_name = 'embedding'


def save_weights(model, weights_dir, name, verbose=True):
	path = os.path.join(weights_dir, name+'.pt')
	validate_path(path, is_dir=False)

	if verbose:
		print(f'saving {name} weights to {path}')

	try:
		torch.save(model.state_dict(), path)
	except Exception:
		raise


def load_weights(model, weights_dir, name, inplace=False, verbose=True):
	path = os.path.join(weights_dir, name+'.pt')

	if not os.path.exists(path):
		raise FileNotFoundError(f'Weights file ({name+".pt"}) was not found in directory ({weights_dir})')

	if verbose:
		print(f'\tloading weights from {path}\n')

	state_dict = torch.load(path)

	if inplace:
		model.load_state_dict(state_dict)
	else:
		loaded_model = deepcopy(model)
		loaded_model.load_state_dict(state_dict)
		return loaded_model


def make_word2vec_model(vocab_size, embed_dim, n_output, layers, device, weights=None, verbose=True):
	if verbose:
		print('Creating Model:')

	model = Word2Vec(
		vocab_size=vocab_size,
		embed_dim=embed_dim,
		n_output=n_output,
		layers=layers
	).to(device)

	if verbose:
		print(model)
		print(f'\tModel contains {sum([p.numel() for p in model.parameters()])} parameters')

	if weights is not None:
		model = load_weights(model, weights, model_name, inplace=False, verbose=verbose)
	else:
		print()


	return model


