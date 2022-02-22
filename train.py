import argparse

import torch
import pytorch_lightning as pl

from gluonts.dataset.repository.datasets import get_dataset
from gluonts.torch.model.deepar.estimator import DeepAREstimator

from utils import get_augmented_dataset

# torch.manual_seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default=None, help='dataset name')
parser.add_argument('--context_length', type=int, default=None, help='model\'s context length')
parser.add_argument('--prediction_length', type=int, default=None, help='model\'s prediction length')
parser.add_argument('--batch_size', type=int, default=128, help='batch size for training')
parser.add_argument('--epochs', type=int, default=50, help='number of training epochs')
parser.add_argument('--perform_augmentation', type=bool, default=False, help='whether to perform randomized training (T/F)')
parser.add_argument('--num_noises', type=int, default=100, help='number of random noises per training series')
parser.add_argument('--sigma', type=float, default=0.1, help='magnitude of noise used for randomized training')

args = parser.parse_args()

dataset = get_dataset(args.dataset, regenerate=False)
batch_size = args.batch_size
epochs = args.epochs
prediction_length = dataset.metadata.prediction_length if args.prediction_length is None else args.prediction_length
context_length = prediction_length * 4 if args.context_length is None else args.context_length
perform_augmentation = args.perform_augmentation
num_noises = args.num_noises
sigma = args.sigma

estimator = DeepAREstimator(
    prediction_length=prediction_length,
    context_length=context_length,
    freq=dataset.metadata.freq,
    batch_size=batch_size,
    trainer_kwargs = {
        'auto_select_gpus': True,
        'gpus': 1 if torch.cuda.is_available() else None,
        'max_epochs': epochs,
    }
)

if perform_augmentation:
    training_data = get_augmented_dataset(
        dataset=dataset,
        num_noises=num_noises,
        sigma=sigma
    )
else:
    training_data = dataset.train

estimator.train(training_data=training_data, cache_data=True)
