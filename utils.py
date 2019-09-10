import os
import yaml
import argparse

import torch


default_config = 'default'
config_dir = 'configs'


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--config', default='default', type=str)

	parser.add_argument('--name', type=str)
	parser.add_argument('--testing', type=bool)
	parser.add_argument('--verbose', type=bool)
	parser.add_argument('--seed', type=int)

	parser.add_argument('--context_window', type=int)
	parser.add_argument('--embed_dim', type=int)
	parser.add_argument('--n_layers', type=int)
	parser.add_argument('--load_dir', type=str)

	parser.add_argument('--batch_size', type=int)
	parser.add_argument('--n_batches', type=int)
	parser.add_argument('--lr', type=float)
	parser.add_argument('--n_epochs', type=int)
	parser.add_argument('--starting_epoch', type=int)

	args = parser.parse_args()
	args = {arg: value for arg, value in vars(args).items() if value is not None}

	return args


def validate_path(path, is_dir=True, verbose=False):
	if not os.path.exists(path):
		if is_dir:
			os.makedirs(path, exist_ok=True)
			if verbose:
				print(f'Created directory {path}\n')
		else:
			head, tail = os.path.split(path)
			os.makedirs(head, exist_ok=True)
			open(path, 'a').close()
			if verbose:
				print(f'Created file {path}\n')


def load_config(config_name):
	config_path = os.path.join(config_dir, config_name + '.yaml')
	with open(config_path, 'r') as fp:
		config = yaml.load(fp, Loader=yaml.FullLoader)
	return config


def save_config(config, path):
	validate_path(path, is_dir=False)
	with open(os.path.join(path, 'config.yaml'), 'w') as fp:
		yaml.dump(config, fp)


def get_args():
	cl_args = parse_args()

	args = load_config(default_config)

	if cl_args['config'] != 'default':
		args.update(load_config(cl_args['config']))

	args.update(cl_args)

	if args['verbose']:
		print('Loading Arguments:')
		for arg, value in args.items():
			print(f'\t{arg}: {value}')
		print()

	return args


def get_device(verbose=True):
	if verbose:
		print('Checking GPU:')
	use_gpu = torch.cuda.is_available()
	if use_gpu:
		device = torch.device('cuda')
		if verbose:
			print('\tUsing GPU\n')
	else:
		device = torch.device('cpu')
		if verbose:
			print('\tCuda is unavailable, using CPU\n')
	return device
