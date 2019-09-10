import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import numpy as np

from data.dataset import NGramDataset
from data.data_utils import load_data
from models.word2vec import Word2Vec


def train(model, generator, opt, criterion):
	model.train()
	losses = []
	batch_size = generator.batch_size
	context_size = generator.dataset.context_size
	for query, context in generator:
		pred = model(query).view(batch_size*context_size, -1)
		context = context.view(-1)
		loss = criterion(pred, context)

		opt.zero_grad()
		loss.backward()
		opt.step()

		losses.append(loss.item())

	return losses



def main():
	data, idx_to_token, token_to_idx, vocab = load_data()
	dataset = NGramDataset(data, context_window=2)
	print(f'Dataset: {len(dataset)} instances, {len(vocab)} vocab size')
	generator = DataLoader(dataset, batch_size=2, shuffle=True)
	model = Word2Vec(len(vocab), embed_dim=20, n_output=dataset.context_size, layers=2)
	print(model)
	opt = torch.optim.SGD(params=model.parameters(), lr=.01)
	criterion = nn.CrossEntropyLoss()

	print('TRAINING')
	loss = train(model, generator, opt, criterion)
	print(loss)


if __name__ == "__main__":
	main()
