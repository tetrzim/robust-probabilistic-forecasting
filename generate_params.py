import argparse
from utils import Params

parser = argparse.ArgumentParser()
parser.add_argument('--n_iterations', type=int, default=200)
parser.add_argument('--learning_rate', type=float, default=0.01)
parser.add_argument('--attack_idx', nargs='+', type=int)
parser.add_argument('--filename', type=str, default=None)
args = parser.parse_args()

params = Params()
params.attack_idx = args.attack_idx
params.n_iterations = args.n_iterations
params.learning_rate = args.learning_rate
params.tolerance = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
params.c = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 300.0, 500.0]
params.modes = ["under", "over"]
params.sigmas = [0.1, 0.2, 0.3, 0.4, 0.5]

if args.filename is not None:
    filename = args.filename
else:
    filename = './attack_params/basic_setup_attack_idx_' + str(params.attack_idx) + '.json'

params.save(filename)
