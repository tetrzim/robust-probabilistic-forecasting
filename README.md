# robust-probabilistic-forecasting

# **Requirements**
Recent versions of GluonTS, PyTorch, and PyTorch Lightning.

# **Model training**
Run train.py.
This file has the following set of command-line arguments:
```console
--dataset:                Name of the dataset
--context_length:         Model's context length
--prediction_length:      Model's prediction length
--batch_size:             Batch size for training
--epochs:                 Number of training epochs
--perform_augmentation:   Whether to perform randomized training (bool)
--num_noises:             Number of random noises per training series
--sigma:                  Magnitude of noise used for randomized training
```

An example command line input
```console
python train.py --dataset electricity_nips --context_length 96 --prediction_length 24 --batch_size 128 --epochs 50 --perform_augmentation True --num_noises 100 --sigma 0.1
```

Keep track of the path within the training log, where the model checkpoint is stored.

# **Adversarial attack**
## **Params file generation**
First run generate_params.py.
This file has command line arguments
```console
--n_iterations:       Number of iterations to compute attack
--learning_rate:      Learning rate used by the optimizer
--attack_idx:         List of attack indices; intergers separated by blank spaces
--filename:           Path to the .json file to be generated
```
Example run:
```console
python generate_params.py --attack_idx X1 X2
```
This will create a .json file under the directory './attack_params/...'\
Then run attack_and_save.py, as in:

```console
python attack_and_save.py --dataset electricity_nips --context_length 96 --prediction_length 24 --model_type baseline --model_path ./lightning_logs/version_X/checkpoints/epoch=XX-step=XXXX.ckpt --device cuda:X --attack_params_path ./attack_params/basic_setup_attack_idx_[X].json
```
Note that this file has command line arguments
```console
--dataset                 Name of the dataset
--context_length:         Model's context length
--prediction_length:      Model's prediction length
--model_type:             Any indicator of the model
--model_path:             Path to model checkpoint
--device:                 Device ('cpu' or 'cuda:X')
--batch_size:             Batch size used for inference
--attack_params_path:     Path to json file containing attack parameters
--num_parallel_samples:   Number of sample paths to compute attacks
```
The '--model_type' argument can be an arbitrary identifier of the model.\
This will create a .pkl file under the directory './attack_results/...' which contains the attack results.

## **Evaluation**
Run evaluate_adversarial.py.
```console
python evaluate_adversarial.py --dataset electricity_nips --context_length 96 --prediction_length 24 --freq H --model_type baseline --model_path ./lightning_logs/version_X/checkpoints/epoch=XX-step=XXXX.ckpt --device cuda:X --attack_params_path ./attack_params/basic_setup_attack_idx_[X].json
```
Note that the command line arguments should be the same as in attack_and_save.py except for
```console
--freq:                  Dataset's frequency
--num_noised_samples:    Number of noised inputs for randomized smoothing 
```
This will create a .pkl file under the directory './metrics/...' which contains the evaluation results.

## **Visualization**
Run visualize_adversarial.py.\
Use the command line arguments
```console
--metric_path_base:      Path to .pkl file containing the Metrics type object (from vanilla model)
--metric_path_rand:      Path to .pkl file containing the Metrics type object (from random-trained model)
--figure_path:           Path to save the figure (and the file name, type)
--criterion:             Metric of interest: should be one of MSE, MAPE, or ND
--sigma_idx:             Index of the column corresponding to smoothed model to plot
--max_tolerance_idx:     Index of the maximum index within the tolerance list to plot
```
Example run:
```console
python visualize_adversarial.py --metric_path_base metrics/XXX.pkl --metric_path_rand metrics/XXX.pkl --figure_path figures/XXX.png --criterion ND --sigma_idx -1 --max_tolerance_idx -3
```

# **Translation by noisy observation**
## **Evaluation**
Run evaluate_translation.py.
```console
--dataset:                Name of the dataset
--context_length:         Model's context length
--prediction_length:      Model's prediction length
--freq:                   Dataset's frequency
--model_type:             Any indicator of the model
--model_path:             Path to model checkpoint
--device:                 Device ('cpu' or 'cuda:X')
--batch_size:             Batch size used for inference
```
Example run:
```console
python evaluate_translation.py --dataset electricity_nips --context_length 96 --prediction_length 24 --freq H --model_type base --model_path ./lightning_logs/version_XX/checkpoints/epoch=XX-step=XXXX.ckpt --device cuda:X
```
This will create a .npy file under the directory './translation_metrics/...'

## **Visualization**
Run visualize_translation.py.\
Use the command line arguments
```console
--table_path_base:       Path to .npy file containing the evaluation results (from vanilla model)
--table_path_rand:       Path to .npy file containing the evaluation results (from random-trained model)
--figure_path:           Path to save the figure (and the file name, type)
--sigma_idx:             Index of the column corresponding to smoothed model to plot
```
Example run:
```console
python visualize_translation.py --table_path_base ./translation_metrics/XXX.npy --table_path_rand ./translation_metrics/XXX.npy --figure_path ./figures/XXX.png
```