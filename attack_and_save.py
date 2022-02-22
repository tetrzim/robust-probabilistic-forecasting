import argparse
import pickle

import numpy as np
import torch
from tqdm.auto import tqdm

from gluonts.dataset.repository.datasets import get_dataset
from gluonts.torch.model.deepar.estimator import DeepAREstimator

from attack_modules.attack import *
from utils import AttackResults, Params
from utils import ts_iter, get_network, get_test_loader
from utils import PREDICTION_INPUT_NAMES

# np.random.seed(0)
# torch.manual_seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default=None, help='dataset name')
parser.add_argument('--context_length', type=int, required=True, help='model\'s context length')
parser.add_argument('--prediction_length', type=int, required=True, help='model\'s prediction length')
parser.add_argument('--model_type', type=str, required=True, help='forecaster type in the form of baseline or RT_sigma')
parser.add_argument('--model_path', type=str, required=True, help='path to model checkpoint')
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='cpu or cuda w/ number specified')
parser.add_argument('--batch_size', type=int, default=128, help='batch size for adversarial attack')
parser.add_argument('--attack_params_path', type=str, required=True, help='path to json file containing attack parameters')
parser.add_argument('--num_parallel_samples', type=int, default=100, help='number of sample paths to compute attacks')
parser.add_argument('--trial_number', type=int, default=None, help='repetitive experiment index for keeping data from multiple trials')

args = parser.parse_args()

dataset_name = args.dataset
dataset = get_dataset(dataset_name, regenerate=False)

prediction_length = args.prediction_length
context_length = args.context_length
device = args.device
batch_size = args.batch_size
num_parallel_samples = args.num_parallel_samples
trial_number = args.trial_number

estimator = DeepAREstimator(
    prediction_length=prediction_length,
    context_length=context_length,
    freq=dataset.metadata.freq,
)

net = get_network(estimator, args.model_path, device)

tss = list(ts_iter(dataset.test, dataset.metadata.freq))

loader = get_test_loader(dataset, estimator, net, device, batch_size)

true_future_targets = [tss[i][0].to_numpy()[-prediction_length:] for i in range(len(tss))]

params = Params(json_path=args.attack_params_path)
params.device = device

filename = "./attack_results/" + args.model_type + "_" + str(prediction_length) + "_" + dataset_name + "_" \
            + "idx_" + str(params.attack_idx)

if trial_number is not None:
    filename = filename + "_" + str(trial_number) + ".pkl"
else:
    filename = filename + ".pkl"

attack = Attack(
    model=net.model,
    params=params,
    input_names=PREDICTION_INPUT_NAMES
)

testset_idx = 0

with open(filename, "wb") as outp:

    for i, batch in tqdm(enumerate(loader)):

        print("Batch ", i)

        targets = batch['past_target']
        batch_size = targets.shape[0]

        future_target = np.array(true_future_targets[testset_idx: testset_idx + batch_size])

        best_perturbation = \
            attack.attack_batch(batch,
                                true_future_target=future_target if device == "cpu"
                                else torch.from_numpy(future_target).float().to(device),
                                num_parallel_samples=num_parallel_samples)

        batch_res = AttackResults(batch=batch,
                                  perturbation=best_perturbation,
                                  true_future_target=future_target,
                                  tolerance=params.tolerance,
                                  attack_idx=params.attack_idx)

        pickle.dump(batch_res, outp, pickle.HIGHEST_PROTOCOL)
        testset_idx += batch_size

        torch.cuda.empty_cache()
