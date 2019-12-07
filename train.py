from torch import nn

import numpy as np
from tqdm import tqdm

from utils import *
from data.data_utils import *
from models.model_utils import *

log = Logger()


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


def train(model, generator, opt, criterion, n_epochs, result_dir, verbose=True):
    model.train()
    losses = []

    for epoch in range(n_epochs):
        log(f'Epoch {epoch}')

        loss = run_epoch(model, generator, opt, criterion)
        losses.append(loss)
        log(f'Loss: {loss}')

        save_weights(model, result_dir, model_name)
        save_weights(model.embedding, result_dir, embedding_name)
        log()

    return losses


def main():
    args = get_args()

    result_dir = os.path.join('results', args['name'])
    validate_path(result_dir, verbose=True and args['verbose'])

    logging_path = os.path.join(result_dir, 'log.txt')
    validate_path(logging_path, is_dir=False, verbose=args['verbose'])
    Logger.set_file_path(logging_path)

    if args['seed'] is not None:
        torch.manual_seed(args['seed'])
        np.random.seed(args['seed'])

    log_path = os.path.join(result_dir, 'log.txt')
    log.init(file_path=log_path, verbosity=int(args['verbose']))

    device = get_device(verbose=args['verbose'])

    save_config(args, result_dir)

    data, idx_to_token, dataset, generator = load_data(
        context_window=args['context_window'],
        batch_size=args['batch_size'],
        device=device,
        testing=args['testing'],
        n_batches=args['n_batches'],
        data_dir=DATA_DIR,
        verbose=args['verbose']
    )
    save_token_set(idx_to_token, result_dir)

    model = make_word2vec_model(
        vocab_size=len(idx_to_token),
        embed_dim=args['embed_dim'],
        n_output=dataset.context_size,
        layers=args['n_layers'],
        device=device,
        weights=args['load_dir'],
        verbose=args['verbose']
    )

    opt = torch.optim.SGD(params=model.parameters(), lr=args['lr'])
    criterion = nn.CrossEntropyLoss()

    log('TRAINING\n')
    losses = train(model, generator, opt, criterion, args['n_epochs'], result_dir)
    log(losses)


if __name__ == "__main__":
    main()
