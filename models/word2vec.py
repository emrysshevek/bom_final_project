import torch
from torch import nn
from torch.nn import functional as F


class Word2Vec(nn.Module):

	def __init__(self, vocab_size, embed_dim, n_output, layers=5, activation=nn.ReLU()):
		super(Word2Vec, self).__init__()
		self.vocab_size = vocab_size
		self.embed_dim = embed_dim
		self.n_output = n_output
		self.output_dim = n_output * vocab_size
		self.activation = activation

		self.embedding = nn.Embedding(vocab_size, embed_dim)
		self.layers = self.make_layers(layers)

	def make_layers(self, layers):
		try:
			int(layers)
		except TypeError:
			layers = nn.ModuleList([nn.Linear(in_dim, out_dim) for in_dim, out_dim in layers])
		else:
			layers = nn.ModuleList(
				[nn.Linear(self.embed_dim if i == 0 else self.output_dim, self.output_dim) for i in range(layers)]
			)
		return layers

	def forward(self, x):
		x = self.embedding(x)
		x = self.activation(x)
		for layer in self.layers:
			x = self.activation(layer(x))
		return x

