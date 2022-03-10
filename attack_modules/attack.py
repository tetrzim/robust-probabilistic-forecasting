import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
from typing import List

import utils


def forward_model(model, inputs, num_parallel_samples=100):

    samples, scale = model(*inputs, num_parallel_samples=num_parallel_samples)
    sample_means = torch.mean(samples, axis=1)
    
    return samples, sample_means, scale


class AttackLoss(nn.Module):

    def __init__(self,
                 c: float,
                 attack_idx: List[int],
                 device="cuda" if torch.cuda.is_available() else "cpu"):
        super(AttackLoss, self).__init__()
        self.c = c
        self.attack_idx = attack_idx
        self.device = device

    def forward(self, perturbation, mu, scale, adv_target):
    
        mu = mu[:, self.attack_idx]
        
        adv_target = adv_target[:, self.attack_idx]

        loss_function = nn.MSELoss(reduction="none")
        
        distance_tensor = loss_function(adv_target / scale, mu / scale)
        distance_per_sample = distance_tensor.sum(axis=1)
        distance = distance_per_sample.sum()

        zero = torch.zeros(perturbation.shape).to(self.device)
        norm_tensor = loss_function(perturbation, zero)
        norm_per_sample = norm_tensor.sum(axis=1)
        norm = norm_per_sample.sum()

        loss_per_sample = norm_per_sample + self.c * distance_per_sample
        loss = loss_per_sample.sum()
        
        return norm_per_sample, distance_per_sample, loss_per_sample, \
               norm, distance, loss


class AttackModule(nn.Module):

    def __init__(self, model, params, c, batch, input_names):

        super(AttackModule, self).__init__()

        self.model = model
        self.params = params
        self.c = c
        self.batch = batch
        self.input_names = input_names
        
        self.attack_loss = AttackLoss(c, attack_idx=self.params.attack_idx, device=self.params.device)
        
        # Initialize perturbation
        self.perturbation = nn.Parameter(torch.zeros_like(batch['past_target']), requires_grad=True)
        self.perturbation.to(self.params.device)

    def generate_adv_target(self, future_target, mode):

        if mode == "over":
            adv_target = 1.5 * future_target
        elif mode == "under":
            adv_target = 0.5 * future_target
        elif mode == "zero":
            adv_target = torch.zeros_like(future_target)
        else:
            raise Exception("No such mode")

        return adv_target

    # Returns predicted mean and scale
    def forward(self, num_parallel_samples=100):
        
        perturbed_inputs = [self.batch[key] * (1 + self.perturbation) if key == 'past_target'
                            else self.batch[key]
                            for key in self.input_names]
   
        _, perturbed_mu, scale = forward_model(self.model, perturbed_inputs, num_parallel_samples=num_parallel_samples)

        return perturbed_mu, scale


class Attack:

    def __init__(self, model, params, input_names):
        self.model = model
        self.params = params
        self.input_names = input_names

        self.model.dropout_rate = 0
        self.model.lagged_rnn.dropout_rate = 0
        self.model.train()
        
        self.max_pert_len = len(params.tolerance)

    def project_perturbation(self, attack_module):

        aux = torch.tensor([-1.], device=self.params.device)

        attack_module.perturbation.data = torch.max(attack_module.perturbation.data, aux)

    def attack_step(self, attack_module, optimizer, adv_target):

        attack_module.zero_grad()

        perturbed_mu, scale = attack_module()

        _, _, _, _, _, loss = \
            attack_module.attack_loss(attack_module.perturbation, perturbed_mu, scale, adv_target)
        
        loss.backward()
        optimizer.step()

        self.project_perturbation(attack_module)

    def attack_batch(self, batch, true_future_target, num_parallel_samples):

        with torch.no_grad():

            inputs = [batch[key] for key in self.input_names]
            _, sample_mu, scale = forward_model(self.model, inputs, num_parallel_samples=num_parallel_samples)
            
            future_target = sample_mu
            
            shape = (self.max_pert_len,) + batch['past_target'].shape
            
            attack_idx = self.params.attack_idx
            
            loss_function = nn.MSELoss(reduction="none")

            # Repeat clean bias max_pert_len times
            best_biases = np.tile(utils.convert_from_tensor(
                loss_function(true_future_target[:, attack_idx] / scale, sample_mu[:, attack_idx] / scale).sum(axis=1)
            ), (self.max_pert_len, 1))
            
            best_perturbation = np.zeros(shape)

        for mode in self.params.modes:

            print("Mode ", mode)

            for i in tqdm(range(len(self.params.c))):

                # Update the lines
                attack_module = AttackModule(model=self.model,
                                             params=self.params,
                                             c=self.params.c[i],
                                             batch=batch,
                                             input_names=self.input_names,)

                adv_target = attack_module.generate_adv_target(future_target=future_target,
                                                               mode=mode)
                
                optimizer = optim.Adam([attack_module.perturbation], lr=self.params.learning_rate)

                # Iterate attack steps
                for _ in tqdm(range(self.params.n_iterations)):
                    self.attack_step(attack_module, optimizer, adv_target)
                    
                # Evaluate the attack
                with torch.no_grad():
                    perturbed_mu, scale = attack_module(num_parallel_samples=num_parallel_samples)

                    c = self.params.c[i]

                    loss = AttackLoss(c, self.params.attack_idx, self.params.device)

                    norm_per_sample, _, _, _, _, _ = \
                        loss(attack_module.perturbation.data,
                             perturbed_mu,
                             scale,
                             adv_target)
                    
                    numpy_norm = np.sqrt(utils.convert_from_tensor(norm_per_sample))
                    numpy_perturbation = utils.convert_from_tensor(attack_module.perturbation.data)
                    
                    numpy_bias = utils.convert_from_tensor(
                        loss_function(true_future_target[:, attack_idx] / scale,
                                      perturbed_mu[:, attack_idx] / scale
                                      ).sum(axis=1)
                    )
                    
                    for l in range(self.max_pert_len):
                        indexes = np.logical_and(numpy_norm <= self.params.tolerance[l],
                                                 numpy_bias > best_biases[l])
                        best_perturbation[l][indexes, :] = numpy_perturbation[indexes, :]
                        best_biases[l][indexes] = numpy_bias[indexes]

        return best_perturbation
