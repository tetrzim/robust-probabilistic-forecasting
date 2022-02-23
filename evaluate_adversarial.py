import argparse
import pickle

import numpy as np
import torch

from tqdm.auto import tqdm

from gluonts.torch.model.deepar.estimator import DeepAREstimator

from utils import Metrics, Params
from utils import load_pickle, change_device, calc_loss, get_network
from utils import PREDICTION_INPUT_NAMES


def smoothed_inference(batch,
                       past_target,
                       net,
                       sigma,
                       device,
                       num_noised_samples: int = 100,
                       intermediate_noise: float = None,
                       retain_positivity: bool = True):

    outputs = []

    for _ in tqdm(range(num_noised_samples), leave=False):
        noised_past_target = change_device(past_target, device)
        noised_past_target += torch.normal(mean=torch.zeros(noised_past_target.shape, device=device),
                                           std=sigma * torch.abs(noised_past_target))
        if retain_positivity:
            noised_past_target = torch.clamp(noised_past_target, min=0)

        noised_inputs = [noised_past_target if key == "past_target"
                         else change_device(batch[key], device) for key in PREDICTION_INPUT_NAMES]

        sample, scale = net.model(*noised_inputs,
                                  num_parallel_samples=1,
                                  intermediate_noise=intermediate_noise)
        outputs.append(sample.detach().cpu().numpy())

        del noised_past_target, noised_inputs, sample, scale
        torch.cuda.empty_cache()

    return np.concatenate(outputs, axis=1)


if __name__ == "__main__":
    # np.random.seed(0)
    # torch.manual_seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=None, help='dataset name')
    parser.add_argument('--context_length', type=int, required=True, help='model\'s context length')
    parser.add_argument('--prediction_length', type=int, required=True, help='model\'s prediction length')
    parser.add_argument('--freq', type=str, default="1H", help='dataset\'s frequency value')
    parser.add_argument('--model_type', type=str, required=True, help='forecaster type, e.g., vanilla, RT, etc.')
    parser.add_argument('--model_path', type=str, required=True, help='path to model checkpoint')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='cpu or cuda w/ number specified')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for adversarial attack')
    parser.add_argument('--attack_params_path', type=str, required=True, help='path to json file containing attack parameters')
    parser.add_argument('--num_parallel_samples', type=int, default=100, help='number of sampling for the baseline model')
    parser.add_argument('--num_noised_samples', type=int, default=100, help='number of noises for randomized smoothing')
    parser.add_argument('--intermediate_noise', type=float, default=None, help='noise magnitude for intermediate future value noising in randomized smoothing')
    parser.add_argument('--trial_number', type=int, default=None, help='repetitive experiment index for keeping data from multiple trials')

    args = parser.parse_args()

    model_path = args.model_path
    prediction_length = args.prediction_length
    context_length = args.context_length
    freq = args.freq
    device = args.device
    num_parallel_samples = args.num_parallel_samples
    num_noised_samples = args.num_noised_samples
    intermediate_noise = args.intermediate_noise
    trial_number = args.trial_number

    params = Params(json_path=args.attack_params_path)

    filename = args.model_type + "_" + str(prediction_length) + "_" + args.dataset + "_" \
                + "idx_" + str(params.attack_idx)
    
    if trial_number is not None:
        filename = filename + "_" + str(trial_number) + ".pkl"
    else:
        filename = filename + ".pkl"

    attack_data = load_pickle("./attack_results/" + filename)
    attack_data = list(attack_data)

    estimator = DeepAREstimator(
        prediction_length=prediction_length,
        context_length=context_length,
        freq=freq,
    )

    net = get_network(estimator, model_path, device)

    tolerance = params.tolerance
    attack_idx = params.attack_idx
    sigmas = params.sigmas

    forecasts = {sigma: {tol: [] for tol in [0.0] + tolerance} for sigma in [0.0] + sigmas}

    for i in tqdm(range(len(attack_data))):

        batch = attack_data[i].batch

        '''
        Clean data, original model
        '''
        inputs = [change_device(batch[key], device) for key in PREDICTION_INPUT_NAMES]
        outputs, scale = net.model(*inputs, num_parallel_samples=num_parallel_samples)
        forecasts[0.0][0.0].append(outputs.detach().cpu().numpy())

        del inputs, outputs, scale
        torch.cuda.empty_cache()

        '''
        Clean data, smoothed model
        '''
        for sigma in tqdm(sigmas, leave=False):
            forecasts[sigma][0].append(
                smoothed_inference(batch, batch["past_target"], net, sigma,
                                   device, num_noised_samples, intermediate_noise)
            )

        for tol in tqdm(tolerance, leave=False):
            '''
            Attacked data, original model
            '''
            perturbation = attack_data[i].perturbation[tolerance.index(tol)]
            perturbation_tensor = change_device(perturbation, device)
            attacked_past_target = change_device(batch["past_target"], device) * (1 + perturbation_tensor)
            attacked_inputs = [attacked_past_target if key == "past_target"
                               else change_device(batch[key], device) for key in PREDICTION_INPUT_NAMES]

            outputs, scale = net.model(*attacked_inputs, num_parallel_samples=num_parallel_samples)
            forecasts[0][tol].append(outputs.detach().cpu().numpy())

            del perturbation_tensor, attacked_inputs, outputs, scale
            torch.cuda.empty_cache()

            '''
            Attacked data, smoothed model
            '''
            for sigma in tqdm(sigmas, leave=False):
                forecasts[sigma][tol].append(
                    smoothed_inference(batch, attacked_past_target, net, sigma,
                                       device, num_noised_samples, intermediate_noise)
                )

    mse, mape, nd, ql = calc_loss(
        attack_data, forecasts, attack_idx=attack_idx, sigmas=sigmas, tolerance=tolerance
    )

    metrics = Metrics(mse=mse, mape=mape, nd=nd, ql=ql, sigmas=sigmas, tolerance=tolerance)

    with open("./metrics/" + filename, "wb") as outp:
        pickle.dump(metrics, outp, pickle.HIGHEST_PROTOCOL)

    # Output test code
    # np.set_printoptions(precision=3)
    # for criterion in ["MSE", "MAPE", "ND"]:
    #     print(criterion)
    #     print(metrics.to_table(criterion))
    #     print("\n")
