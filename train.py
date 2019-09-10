import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import numpy as np
import argparse
import os
from tqdm import tqdm

from data.dataset import NGramDataset
from data.data_utils import load_data
from models.word2vec import Word2Vec


def run_epoch(model, generator, opt, criterion):
    batch_size = generator.batch_size
    context_size = generator.dataset.context_size
    losses = []
    for query, context in tqdm(generator):
        pred = model(query).view(batch_size * context_size, -1)
        context = context.view(-1)
        loss = criterion(pred, context)

        opt.zero_grad()
        loss.backward()
        opt.step()

        loss = loss.detach().cpu().item()
        losses.append(loss)

    return np.mean(losses)


def train(model, generator, opt, criterion, n_epochs, model_name):
    model.train()
    losses = []
    result_dir = os.path.join('weights')

    for epoch in range(n_epochs):
        print(f'Epoch {epoch}')
        loss = run_epoch(model, generator, opt, criterion)
        losses.append(loss)
        print(f'Loss: {loss}')

        print('Saving model')
        torch.save(model, os.path.join('weights', model_name + '_model.pt'))

        print('Saving embedding')
        torch.save(model.embedding, os.path.join('weights', model_name + '_embedding.pt'))

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
    generator = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    print(f'DATA: {len(dataset)} instances, {len(vocab)} vocab size, {len(generator)} batches')

    model = Word2Vec(len(vocab), embed_dim=args.embed_dim, n_output=dataset.context_size, layers=2).to(device)
    print(model)
    print(f'Model contains {sum([p.numel() for p in model.parameters()])} parameters')

    opt = torch.optim.SGD(params=model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    print('TRAINING')
    losses = train(model, generator, opt, criterion, args.n_epochs, args.name)
    print(losses)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--context_window', default=5, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--embed_dim', default=256, type=int)
    parser.add_argument('--n_layers', default=5, type=int)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--n_epochs', default=100, type=int)
    parser.add_argument('--name', default='sample', type=str)
    args = parser.parse_args()

    main(args)
