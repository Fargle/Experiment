import argparse
import errno
import json
import os
import sys
from typing import List

import pandas as pd
import wandb
from attr import dataclass
from wandb.keras import WandbCallback

from model.model import Model

# def verbose()


class WB:
    """
    Weights and Biases api connection.

    config: A dictionary of model parameters.
    project: WB project name, specified in experiment_config.json.
    entity: Wb entity, specified in experiment_config.json.
    """

    def __init__(self, config):
        """Initialize Weights and Biases"""
        wandb.init(project="nalu-2021", entity="nalu-ai")
        wandb.config.update(config)
        self.callbacks = WandbCallback()


@dataclass
class Experiment:
    """Defines an Experiment"""

    X_train: pd.DataFrame
    y_train: pd.DataFrame
    X_test: pd.DataFrame
    y_test: pd.DataFrame
    model: Model

    def conduct(self):
        """Conduct experiment."""
        self.model.init()
        self.model.fit(self.X_train, self.y_train)


def save(experiment: Experiment, name):
    """Save an experiment"""
    experiment.model.save(f"{name}/model")


def load_config(path):
    """Load config"""
    with open(path, "r", encoding="utf-8") as cnfg:
        config = json.load(cnfg)

    return config


def load_train_test_set(path: str):
    """Load the train/test data"""
    data_to_gather = ["train_X", "test_X", "train_y", "test_y"]
    data = [pd.read_csv(f"{path}/{data}.csv") for data in data_to_gather]
    return data


def get_arguments():
    "Get command line arguments"

    parser = argparse.ArgumentParser("Conduct Experiment")
    parser.add_argument(
        "-experiment",
        "-p",
        type=str,
        default=None,
        help="Path to train/test data directory.",
    )
    return parser.parse_args()


def main():
    """Main"""

    arguments = get_arguments()
    exp_name = arguments.experiment

    # load model
    model_config = load_config(f"{exp_name}/model_config.json")
    model = Model(model_config)

    # load data
    data = load_train_test_set(f"{exp_name}")

    experiment = Experiment(
        X_train=data[0], X_test=data[1], y_train=data[2], y_test=data[3], model=model
    )

    experiment.conduct()
    save(experiment, exp_name)


if __name__ == "__main__":
    main()
