import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import numpy as np
import argparse

from data.dataset import NGramDataset
from data.data_utils import load_data
from models.word2vec import Word2Vec


def run_epoch(model, generator, opt, criterion):
	batch_size = generator.batch_size
	context_size = generator.dataset.context_size
	losses = []
	for query, context in generator:
		pred = model(query).view(batch_size * context_size, -1)
		context = context.view(-1)
		loss = criterion(pred, context)

		opt.zero_grad()
		loss.backward()
		opt.step()

		loss = loss.detach().cpu().item()
		losses.append(loss)
		print(loss)

	return np.mean(losses)


def train(model, generator, opt, criterion, n_epochs):
	model.train()
	losses = []

	for epoch in range(n_epochs):
		loss = run_epoch(model, generator, opt, criterion)
		losses.append(loss)

	return losses


def main(args):
	print(args)

	use_gpu = torch.cuda.is_available()
	if use_gpu:
		device = torch.device('cuda')
		print('Using GPU')
	else:
		device = torch.device('cpu')
		print('Cuda is unavailable, using CPU')

	data, idx_to_token, token_to_idx, vocab = load_data()
	dataset = NGramDataset(data, context_window=args.context_window, device=device)
	generator = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
	print(f'DATA: {len(dataset)} instances, {len(vocab)} vocab size, {len(generator)} batches')

	model = Word2Vec(len(vocab), embed_dim=args.embed_dim, n_output=dataset.context_size, layers=2).to(device)
	print(model)

	opt = torch.optim.SGD(params=model.parameters(), lr=args.lr)
	criterion = nn.CrossEntropyLoss()

	print('TRAINING')
	losses = train(model, generator, opt, criterion, args.n_epochs)
	print(losses)


if __name__ == "__main__":
	argparser = argparse.ArgumentParser()
	argparser.add_argument('--context_window', default=5, type=int)
	argparser.add_argument('--batch_size', default=8, type=int)
	argparser.add_argument('--embed_dim', default=100, type=int)
	argparser.add_argument('--n_layers', default=5, type=int)
	argparser.add_argument('--lr', default=0.01, type=float)
	argparser.add_argument('--n_epochs', default=10, type=int)
	args = argparser.parse_args()

	main(args)
