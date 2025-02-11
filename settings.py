import torch
import yaml

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)['config']

if config is None:
    config = {}

config['device'] = torch.device(
    "cuda" if torch.cuda.is_available() else
    # "mps" if torch.mps.is_available() else
    "cpu"
)
