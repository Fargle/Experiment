from typing import List

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import activations, layers, optimizers

from model.custom_callbacks import CustomCallback


class Model:
    """Sequential Model"""

    def __init__(self, config: dict):
        """Main init"""

        self.model_init = config["model_init"]
        self.model_fit = config["model_fit"]

        self.loss = self.model_init["loss"]
        self.activation = getattr(activations, self.model_init["activation"])
        self.optimizer = getattr(optimizers, self.model_init["optimizer"])(
            self.model_init["learning_rate"]
        )

        self.model = None

    def get_layers(self, dims: List, activation):
        """Get dense layers."""
        return [layers.Dense(dim, activation) for dim in dims]

    def init(self):
        """Initialize model and compile."""

        dense_layers = [
            layers.Dense(dim, activation=self.activation)
            for dim in self.model_init["dims"]
        ]
        model = keras.Sequential(layers=dense_layers)
        model.compile(loss=self.loss, optimizer=self.optimizer)
        self.model = model

    def fit(self, X, y):
        """fit the model to data"""
        self.model.fit(X, y, **self.model_fit)

    def save(self, path: str):
        """Save model"""
        self.model.save(path)
