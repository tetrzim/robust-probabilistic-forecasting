import json
import pickle
import random

import torch
import numpy as np

import pandas as pd
from tqdm.auto import tqdm

from typing import Iterator, Optional
from gluonts.dataset.common import DataEntry, Dataset, ListDataset
from gluonts.dataset.loader import InferenceDataLoader
from gluonts.torch.batchify import batchify
from gluonts.torch.model.deepar.lightning_module import DeepARLightningModule
from gluonts.transform import AdhocTransform

from mymodule.module import DeepARModel
from mymodule.FSmodule import DeepARModel_FS

PREDICTION_INPUT_NAMES = [
    "feat_static_cat",
    "feat_static_real",
    "past_time_feat",
    "past_target",
    "past_observed_values",
    "future_time_feat",
]


class Params:
    '''Class that loads hyperparameters from a json file.
    Example:
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    '''

    def __init__(self, json_path: str = None):
        if json_path is not None:
            with open(json_path) as f:
                params = json.load(f)
                self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4, ensure_ascii=False)

    def update(self, json_path):
        '''Loads parameters from json file'''
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        '''Gives dict-like access to Params instance by params.dict['learning_rate']'''
        return self.__dict__


class AttackResults:

    def __init__(self,
                 batch,
                 perturbation,
                 true_future_target,
                 tolerance,
                 attack_idx
                 ):
        self.batch = batch
        self.perturbation = perturbation
        self.true_future_target = true_future_target
        self.tolerance = tolerance
        self.attack_idx = attack_idx


class Metrics:

    def __init__(self, mse, mape, nd, ql, sigmas, tolerance, quantiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]):
        self.mse = mse
        self.mape = mape
        self.nd = nd
        self.ql = ql
        self.sigmas = sigmas
        self.tolerance = tolerance
        self.quantiles = quantiles

    def to_table(self, criterion: str):

        assert criterion in ["MSE", "MAPE", "ND", "QL"], "Invalid criterion"

        if criterion == "QL":
            table = []
            for tol in [0.0] + self.tolerance:
                table.append(np.array([np.average(self.ql[sigma][tol]) for sigma in [0.0] + self.sigmas]))
            return np.array(table)

        def avg_columns(arr):
            return np.reshape(np.average(arr, axis=1), (arr.shape[0], 1))

        def sum_columns(arr):
            return np.reshape(np.sum(arr, axis=1), (arr.shape[0], 1))

        if criterion == "MSE":
            data = self.mse
            func1 = avg_columns
            func2 = np.average
        elif criterion == "MAPE":
            data = self.mape
            func1 = avg_columns
            func2 = np.average
        else:
            data = self.nd
            func1 = sum_columns
            func2 = np.sum

        table = []
        for tol in [0.0] + self.tolerance:
            row = np.hstack([func1(data[sigma][tol]) for sigma in [0.0] + self.sigmas])
            row = row[np.isfinite(row)].reshape(-1, row.shape[1])
            table.append(func2(np.abs(row), axis=0))

        return np.vstack(table)


def get_augmented_dataset(dataset, num_noises: int=100, sigma: float=0.1):

    train_data_list = list(iter(dataset.train))
    train_length = len(train_data_list)
    
    for _ in range(num_noises):
        for idx in range(train_length):
            target = train_data_list[idx]['target']
            data = {'start': train_data_list[idx]['start'],
                    'target': target + np.random.normal(loc=np.zeros_like(target),
                                                        scale=sigma * target),
                    'feat_static_cat': train_data_list[idx]['feat_static_cat'].copy(),
                    'item_id': None,
                    'source': None,
                }
            train_data_list.append(data)

    random.shuffle(train_data_list)
    return ListDataset(data_iter=train_data_list, freq=dataset.metadata.freq)
    

def get_network(estimator, model_path, device):

    model = DeepARModel(
        freq=estimator.freq,
        context_length=estimator.context_length,
        prediction_length=estimator.prediction_length,
        num_feat_dynamic_real=(
                1 + estimator.num_feat_dynamic_real + len(estimator.time_features)
        ),
        num_feat_static_real=max(1, estimator.num_feat_static_real),
        num_feat_static_cat=max(1, estimator.num_feat_static_cat),
        cardinality=estimator.cardinality,
        embedding_dimension=estimator.embedding_dimension,
        num_layers=estimator.num_layers,
        hidden_size=estimator.hidden_size,
        distr_output=estimator.distr_output,
        dropout_rate=estimator.dropout_rate,
        lags_seq=estimator.lags_seq,
        scaling=estimator.scaling,
        num_parallel_samples=estimator.num_parallel_samples,
    )

    net = DeepARLightningModule(model=model, loss=estimator.loss)
    checkpoint = torch.load(model_path)
    net.load_state_dict(checkpoint['state_dict'])
    net.to(device)

    return net

def get_FS_network(estimator, model_path, device):

    model = DeepARModel_FS(
        freq=estimator.freq,
        context_length=estimator.context_length,
        prediction_length=estimator.prediction_length,
        num_feat_dynamic_real=(
                1 + estimator.num_feat_dynamic_real + len(estimator.time_features)
        ),
        num_feat_static_real=max(1, estimator.num_feat_static_real),
        num_feat_static_cat=max(1, estimator.num_feat_static_cat),
        cardinality=estimator.cardinality,
        embedding_dimension=estimator.embedding_dimension,
        num_layers=estimator.num_layers,
        hidden_size=estimator.hidden_size,
        distr_output=estimator.distr_output,
        dropout_rate=estimator.dropout_rate,
        lags_seq=estimator.lags_seq,
        scaling=estimator.scaling,
        num_parallel_samples=estimator.num_parallel_samples,
    )

    net = DeepARLightningModule(model=model, loss=estimator.loss)
    checkpoint = torch.load(model_path)
    net.load_state_dict(checkpoint['state_dict'])
    net.to(device)

    return net

def get_test_loader(dataset, estimator, net, device, batch_size: int = 128, lead_time: Optional[int] = 0):

    prediction_length = estimator.prediction_length

    def truncate_target(data):
        data = data.copy()
        target = data["target"]
        assert (
                target.shape[-1] >= prediction_length
        )  # handles multivariate case (target_dim, history_length)
        data["target"] = target[..., : -prediction_length - lead_time]
        return data

    dataset_test_trunc = AdhocTransform(truncate_target).apply(dataset.test)

    return InferenceDataLoader(
        dataset_test_trunc,
        transform=estimator.create_transformation() + estimator._create_instance_splitter(net, "test"),
        batch_size=batch_size,
        stack_fn=lambda data: batchify(data, device),
    )


def convert_from_tensor(var):
    if isinstance(var, torch.Tensor):
        var = var.cpu().numpy()
    return var


def add_ts_dataframe(
    data_iterator: Iterator[DataEntry], freq
) -> Iterator[DataEntry]:
    for data_entry in data_iterator:
        data = data_entry.copy()
        index = pd.date_range(
            start=data["start"],
            freq=freq,
            periods=data["target"].shape[-1],
        )
        data["ts"] = pd.DataFrame(
            index=index, data=data["target"].transpose()
        )
        yield data


def ts_iter(dataset: Dataset, freq) -> pd.DataFrame:
    for data_entry in add_ts_dataframe(iter(dataset), freq):
        yield data_entry["ts"]


def load_pickle(filename):
    with open(filename, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break


def change_device(var, device):

    if isinstance(var, torch.Tensor):
        if var.device == "cpu":
            var.to(device)
            return var
        elif var.device != device:
            return var.cpu().to(device)
        else:
            return var

    return torch.from_numpy(var).float().to(device)


def calc_loss(attack_data, forecasts, attack_idx, sigmas, tolerance, quantiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]):

    testset_size = sum([attack_data[i].true_future_target.shape[0] for i in range(len(attack_data))])

    mse = {sigma: {tol: np.zeros((testset_size, len(attack_idx))) for tol in [0] + tolerance}
                  for sigma in [0] + sigmas}
    mape = {sigma: {tol: np.zeros((testset_size, len(attack_idx))) for tol in [0] + tolerance}
              for sigma in [0] + sigmas}
    nd = {sigma: {tol: np.zeros((testset_size, len(attack_idx))) for tol in [0] + tolerance}
                 for sigma in [0] + sigmas}
    ql = {sigma: {tol: np.zeros((len(quantiles), testset_size, len(attack_idx))) for tol in [0] + tolerance}
                 for sigma in [0] + sigmas}

    nd_denom = sum([np.sum(attack_data[i].true_future_target[:, attack_idx])
                    for i in range(len(attack_data))])

    testset_idx = 0

    for i in tqdm(range(len(attack_data))):

        true_future_target = attack_data[i].true_future_target
        batch_size = true_future_target.shape[0]

        for sigma in [0] + sigmas:
            for tol in [0] + tolerance:
                mape[sigma][tol][testset_idx: testset_idx + batch_size] = \
                    np.average(np.transpose(forecasts[sigma][tol][i][:, :, attack_idx],
                                            (1, 0, 2)), axis=0) \
                    / true_future_target[:, attack_idx] - 1.

                mse[sigma][tol][testset_idx: testset_idx + batch_size] = \
                    mape[sigma][tol][testset_idx: testset_idx + batch_size] ** 2

                nd[sigma][tol][testset_idx: testset_idx + batch_size] = \
                    np.abs(
                        np.average(np.transpose(
                            forecasts[sigma][tol][i][:, :, attack_idx], (1, 0, 2)) \
                                   - true_future_target[:, attack_idx], axis=0)) / nd_denom
                
                quantile_forecasts = np.quantile(
                    a=forecasts[sigma][tol][i][:, :, attack_idx],
                    q=quantiles,
                    axis=1
                )

                ql[sigma][tol][:, testset_idx: testset_idx + batch_size] = \
                    np.maximum(
                        np.array(quantiles) * (quantile_forecasts - true_future_target[:, attack_idx]).T,
                        (np.array(quantiles) - 1) * (quantile_forecasts - true_future_target[:, attack_idx]).T
                    ).T

        testset_idx += batch_size

    return mse, mape, nd, ql
