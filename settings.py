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


def get_args(args: list[str]):
    step_by_step = any(arg == '--step' for arg in args)
    manual = any(arg == '--manual' for arg in args)
    train = any(arg == '--train' for arg in args) and not manual
    fps = 60 if not manual else 0
    load_model = \
        next((arg for arg in args if arg.startswith('--model=')), None)\
        if not manual else None
    load_model = load_model.split('=')[1] \
        if load_model is not None else None
    episodes = \
        next((arg for arg in args if arg.startswith('--episodes=')), None)\
        if train else None
    episodes = int(episodes.split('=')[1]) \
        if episodes is not None else None

    return step_by_step, manual, train, fps, load_model, episodes
