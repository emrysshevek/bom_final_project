import torch
from torch.utils.data import Dataset


class NGramDataset(Dataset):

	def __init__(self, tokens, context_window=5, device=torch.device('cpu')):
		self.tokens = tokens
		self._make_context_window(context_window)
		self.device = device

		self.context_key = 'context'
		self.query_key = 'query'
		self.n_grams = []
		self.bag_tokens()

	def _make_context_window(self, context_window):
		try:
			int(context_window)
		except TypeError:
			self.n_left, self.n_right = context_window
		else:
			self.n_left, self.n_right = (context_window, context_window)
		finally:
			self.context_size = self.n_left + self.n_right

	def bag_tokens(self):
		for i, token in enumerate(self.tokens[self.n_left:-self.n_right]):
			context = self.tokens[i - self.n_left:i] + self.tokens[i + 1:i + self.n_right + 1]
			self.n_grams.append({
				self.context_key: context,
				self.query_key: token})

	def __len__(self):
		return len(self.n_grams)

	def __getitem__(self, idx):
		item = self.n_grams[idx]
		context = torch.tensor(item[self.context_key], device=self.device, dtype=torch.long)
		query = torch.tensor(item[self.query_key], device=self.device, dtype=torch.long)
		return query, context
