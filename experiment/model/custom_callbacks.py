import wandb
from tensorflow import keras


class CustomCallback(keras.callbacks.Callback):
    """keras custom callback."""

    def on_epoch_end(self, epoch, logs=None):
        """log validation loss."""
        wandb.log({"validation_loss": logs.get("val_loss")})
