import argparse

import numpy as np
import torch
from tqdm.auto import tqdm

from gluonts.dataset.field_names import FieldName
from gluonts.dataset.loader import InferenceDataLoader
from gluonts.dataset.repository.datasets import get_dataset
from gluonts.torch.batchify import batchify
from gluonts.torch.model.deepar.estimator import DeepAREstimator
from gluonts.transform import InstanceSplitter, TestSplitSampler

from utils import PREDICTION_INPUT_NAMES, change_device, get_FS_network, ts_iter


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=None, help='dataset name')
    parser.add_argument('--context_length', type=int, required=True, help='model\'s context length')
    parser.add_argument('--prediction_length', type=int, required=True, help='model\'s prediction length')
    parser.add_argument('--model_type', type=str, required=True, help='forecaster type in the form of baseline or RT_sigma')
    parser.add_argument('--model_path', type=str, required=True, help='path to model checkpoint')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='cpu or cuda w/ number specified')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for adversarial attack')
    parser.add_argument('--trial_number', type=int, required=True, help='repetitive experiment index for keeping data from multiple trials')

    args = parser.parse_args()

    dataset_name = args.dataset
    dataset = get_dataset(dataset_name, regenerate=False)

    prediction_length = args.prediction_length
    context_length = args.context_length
    device = args.device
    batch_size = args.batch_size

    estimator = DeepAREstimator(
        prediction_length=prediction_length,
        context_length=context_length,
        freq=dataset.metadata.freq,
    )

    net = get_FS_network(estimator, args.model_path, device)

    instance_sampler = TestSplitSampler()

    instance_splitter = InstanceSplitter(
        target_field=FieldName.TARGET,
        is_pad_field=FieldName.IS_PAD,
        start_field=FieldName.START,
        forecast_start_field=FieldName.FORECAST_START,
        instance_sampler=instance_sampler,
        past_length=net.model._past_length + prediction_length + 3,
        future_length=prediction_length,
        time_series_fields=[
            FieldName.FEAT_TIME,
            FieldName.OBSERVED_VALUES,
        ],
        dummy_value=estimator.distr_output.value_in_support
    )

    loader = InferenceDataLoader(
        dataset.test,
        transform=estimator.create_transformation() + instance_splitter,
        batch_size=batch_size,
        stack_fn=lambda data: batchify(data, device),
    )

    tss = list(ts_iter(dataset.test, dataset.metadata.freq))

    perturbation_levels = [-0.9, -0.8, -0.5, -0.2, -0.1, 0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0, 9.0]
    sigmas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    num_parallel_samples = 100
    num_noised_samples = 100

    original_forecasts = {sigma: [] for sigma in [0.0] + sigmas}
    translated_forecasts = {pert: {sigma: [] for sigma in [0.0] + sigmas} for pert in perturbation_levels}
    
    num_batches = 0

    for i, batch in tqdm(enumerate(loader), leave=True):        
        
        print("Batch ", i)
        batch_size = batch['past_target'].shape[0]

        for key in PREDICTION_INPUT_NAMES:
            batch[key] = change_device(batch[key], device)

        ### Basic/smoothed inference on original inputs
        original_inputs = []
        for key in PREDICTION_INPUT_NAMES:
            if key == 'past_target' or key == 'past_observed_values':
                original_inputs.append(batch[key][:, :-prediction_length - 1])
            elif key == 'past_time_feat':
                original_inputs.append(batch[key][:, :-prediction_length - 1, :])
            elif key == 'future_time_feat':
                original_inputs.append(batch['past_time_feat'][:, -prediction_length - 1: -1, :])
            else:
                original_inputs.append(batch[key])
        
        outputs, _ = net.model(*original_inputs, num_parallel_samples=num_parallel_samples)
        original_forecasts[0.0].append(outputs.detach().cpu().numpy())

        del outputs
        torch.cuda.empty_cache()

        for sigma in sigmas:
            outputs, _ = net.model(*original_inputs, num_parallel_samples=num_parallel_samples, intermediate_noise=sigma)
            original_forecasts[sigma].append(
                outputs.detach().cpu().numpy()
            ) 
            del outputs
            torch.cuda.empty_cache()

        del original_inputs
        torch.cuda.empty_cache()

        ### Basic/smoothed inference on translated inputs
        for pert in perturbation_levels:
            
            ### Perturb the "latest" index
            pert_idx = [-prediction_length - 1]
            original_vals = batch['past_target'][:, pert_idx].clone()
            batch['past_target'][:, pert_idx] = (1 + pert) * original_vals
            
            translated_inputs = []
            for key in PREDICTION_INPUT_NAMES:

                if key == 'past_target' or key == 'past_observed_values':
                    translated_inputs.append(batch[key][:, 1: -prediction_length])
                elif key == 'past_time_feat':
                    translated_inputs.append(batch[key][:, 1: -prediction_length, :])
                elif key == 'future_time_feat':
                    translated_inputs.append(batch['past_time_feat'][:, -prediction_length:, :])
                else:
                    translated_inputs.append(batch[key])
                    
            # translated_outputs, _ = net.model(*translated_inputs, num_parallel_samples=num_parallel_samples,)
            translated_outputs, _ = net.model(*translated_inputs, num_parallel_samples=num_parallel_samples, noise_lag_index=1)
            translated_forecasts[pert][0.0].append(translated_outputs.detach().cpu().numpy())

            del translated_outputs
            torch.cuda.empty_cache()

            for sigma in sigmas:
                # outputs, _ = net.model(*translated_inputs, num_parallel_samples=num_parallel_samples, intermediate_noise=sigma)
                outputs, _ = net.model(*translated_inputs, num_parallel_samples=num_parallel_samples, intermediate_noise=sigma, noise_lag_index=1)
                translated_forecasts[pert][sigma].append(
                    outputs.detach().cpu().numpy()
                )
                del outputs
                torch.cuda.empty_cache()
            
            ### Restore the original "latest" index
            batch['past_target'][:, pert_idx] = original_vals
            
            del original_vals, translated_inputs
            torch.cuda.empty_cache()
        
        num_batches += 1

        break

    table = []

    for pert in perturbation_levels:

        row = []
        
        for sigma in [0.0] + sigmas:
            
            denom = sum([np.sum(np.abs(np.average(original_forecasts[sigma][i][:, :, 1:], axis=1)))
                for i in range(num_batches)])

            row.append(sum([
                np.sum(
                    np.abs(
                        np.average(original_forecasts[sigma][i][:, :, 1:], axis=1) - \
                        np.average(translated_forecasts[pert][sigma][i][:, :, :-1], axis=1)
                        # np.median(original_forecasts[sigma][i][:, :, 1:], axis=1) - \
                        # np.median(translated_forecasts[pert][sigma][i][:, :, :-1], axis=1)
                    )
                )
                for i in range(num_batches)
            ]) / denom)
        
        table.append(np.array(row))

    table = np.array(table)
    np.set_printoptions(precision=3)
    print(table)

    np.save(
        file='./translation_metrics/' + dataset_name + '_' + str(prediction_length) + '_' + args.model_type + '_translation_test_' \
                + str(args.trial_number) + '.npy',
        arr=table
    )
