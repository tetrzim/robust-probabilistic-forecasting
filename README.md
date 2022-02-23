# **robust-probabilistic-forecasting**
This is the public repo for the paper "Robust Probabilistic Time Series Forecasting" (AISTATS '22).

# **Requirements**
Recent versions of GluonTS, PyTorch, and PyTorch Lightning.

# **Datasets**
Use the following 4 dataset names and the corresponding specified parameters to reproduce the results of the paper.
```console
exchange_rate_nips      prediction_length = 30
                        context_length = 120
                        freq = 'B'

m4_daily                prediction_length = 14
                        context_length = 56
                        freq = 'D'

traffic_nips            prediction_length = 24
                        context_length = 96
                        freq = 'H'

electricity_nips        prediction_length = 24
                        context_length = 96
                        freq = 'H'
```

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
python train.py --dataset exchange_rate_nips --context_length 120 --prediction_length 30 --batch_size 128 --epochs 50 --perform_augmentation True --num_noises 100 --sigma 0.1
```

Keep track of the path within the training log, where the model checkpoint is stored.

# **Additive adversarial attack**
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
python generate_params.py --attack_idx -1
```
This will create a .json file under the directory './attack_params/...'.
## **Perform adversarial attack and save the results**
Then run attack_and_save.py, as in:
```console
python attack_and_save.py --dataset exchange_rate_nips --context_length 120 --prediction_length 30 --model_type vanilla --model_path ./lightning_logs/version_X/checkpoints/epoch=XX-step=XXXX.ckpt --device cuda:0 --attack_params_path ./attack_params/basic_setup_attack_idx_[-1].json
```
Note that this file has command line arguments
```console
--dataset                 Name of the dataset
--context_length:         Model's context length
--prediction_length:      Model's prediction length
--model_type:             An indicator of the forecaster type, e.g., vanilla, RT, etc.
--model_path:             Path to model checkpoint
--device:                 Device ('cpu' or 'cuda:X')
--batch_size:             Batch size used for inference
--attack_params_path:     Path to json file containing attack parameters
--num_parallel_samples:   Number of sample paths used to perform adversarial attack
```
This will create a .pkl file under the directory './attack_results/...' which contains the attack results.

## **Evaluation**
Run evaluate_adversarial.py.\
This file has the same command line arguments as 'attack_and_save.py', plus:
```console
--freq:                   The frequency type ('B', 'D', 'H', etc.) of the dataset
--num_noised_samples:     Number of sample paths used to perform smoothed inference (randomized smoothing)
```
Example run:
```console
python evaluate_adversarial.py --dataset exchange_rate_nips --context_length 120 --prediction_length 30 --freq B --model_type vanilla --model_path ./lightning_logs/version_X/checkpoints/epoch=XX-step=XXXX.ckpt --device cuda:0 --attack_params_path ./attack_params/basic_setup_attack_idx_[-1].json
```
This will create a .pkl file under the directory './metrics/...' which contains the evaluation results.

## **Visualization**
Run visualize_adversarial.py.\
Use the command line arguments
```console
--metric_path_base:      Path to .pkl file containing the Metrics type object (from vanilla model)
--metric_path_rand:      Path to .pkl file containing the Metrics type object (from random-trained model)
--figure_path:           Path to save the figure to
--criterion:             Metric of interest: should be one of MSE, MAPE, or ND (default is ND)
--sigma_idx:             Index of the column corresponding to the desired value of smoothing variance
--max_tolerance_idx:     Index of the maximum index within the tolerance list to plot
```
Example run:
```console
python visualize_adversarial.py --metric_path_base metrics/XXX.pkl --metric_path_rand metrics/XXX.pkl --figure_path figures/XXX.png --criterion ND --sigma_idx -1 --max_tolerance_idx -1
```

# **Time shift with noisy observation**
## **Evaluation**
Run evaluate_translation.py.\
This file has the following set of command-line arguments:
```console
--dataset:                Name of the dataset
--context_length:         Model's context length
--prediction_length:      Model's prediction length
--model_type:             Any indicator of the model
--model_path:             Path to model checkpoint
--device:                 Device ('cpu' or 'cuda:X')
--batch_size:             Batch size used in inference (forecast generation)
```
Example run:
```console
python evaluate_translation.py --dataset m4_daily --context_length 56 --prediction_length 14 --model_type vanilla --model_path ./lightning_logs/version_XX/checkpoints/epoch=XX-step=XXXX.ckpt --device cuda:0
```
This will create a .npy file under the directory './translation_metrics/...'

## **Visualization**
Run visualize_translation.py.\
Use the command line arguments
```console
--table_path_base:       Path to .npy file containing the evaluation results (from vanilla model)
--table_path_rand:       Path to .npy file containing the evaluation results (from random-trained model)
--figure_path:           Path to save the figure to
--sigma_idx:             Index of the column corresponding to the desired value of smoothing variance
```
Example run:
```console
python visualize_translation.py --table_path_base ./translation_metrics/XXX.npy --table_path_rand ./translation_metrics/XXX.npy --figure_path ./figures/XXX.png
```