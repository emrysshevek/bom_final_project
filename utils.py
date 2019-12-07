import os
import yaml
import argparse

import torch


default_config = 'default'
config_dir = 'configs'


class Logger:

	verbosity = 0
	file_path = None

	@classmethod
	def init(cls, file_path=None, verbosity=None):
		if verbosity is not None:
			cls.set_verbosity(verbosity)
		if file_path is not None:
			cls.set_file_path(file_path)

	@classmethod
	def set_file_path(cls, file_path):
		Logger.file_path = file_path
		validate_path(cls.file_path, is_dir=False)
		open(cls.file_path, 'w').close()

	@classmethod
	def set_verbosity(cls, verbosity):
		cls.verbosity = verbosity

	def __call__(self, *args, **kwargs):
		verbosity = kwargs.pop('verbosity', 0)

		if self.verbosity:
			print(*args, **kwargs)
		if self.file_path is not None:
			with open(self.file_path, 'a') as fp:
				kwargs['file'] = fp
				print(*args, **kwargs)


log = Logger()


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
			log(f'Created directory {path}\n')
		else:
			head, tail = os.path.split(path)
			os.makedirs(head, exist_ok=True)
			open(path, 'a').close()
			log(f'Created file {path}\n')


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

	Logger.init(file_path=None, verbosity=int(args['verbose']))

	log('Loading Arguments:')
	for arg, value in args.items():
		log(f'\t{arg}: {value}')
	log()

	return args


def get_device(verbose=True):
	log('Checking GPU:')
	use_gpu = torch.cuda.is_available()
	if use_gpu:
		device = torch.device('cuda')
		log('\tUsing GPU\n')
	else:
		device = torch.device('cpu')
		log('\tCuda is unavailable, using CPU\n')
	return device

