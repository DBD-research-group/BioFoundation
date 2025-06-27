import os
import torch
import numpy as np
from lightning.pytorch.callbacks import Callback
from biofoundation.modules.models.biofoundation_model import BioFoundationModel


class SpectrogramSaver(Callback):
    def __init__(
        self,
        output_dir: str,
        every_n_batches: int = 1,
        amount: int = 10,
        dataset: str = "HSN",
    ):
        """
        output_dir: Directory to save spectrogram .npy files.
        every_n_batches: Save spectrograms every N batches.
        amount: Number of spectrograms to save.
        """
        super().__init__()
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        print("Directory exists?", os.path.isdir(self.output_dir))
        print("Absolute path:", os.path.abspath(self.output_dir))
        self.every_n_batches = every_n_batches
        self.amount = amount
        self.counter = 0
        self.dataset = dataset

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % self.every_n_batches != 0:
            return
        if self.counter < self.amount:
            model = pl_module.model
            preprocessed = model._preprocess(batch["input_values"])
            specs = preprocessed.cpu().numpy()
            np.save(
                self.output_dir
                + f"/spectrograms_batch{model.__class__.__name__}-{self.dataset}-{batch_idx}.npy",
                specs,
            )
            self.counter += 1
            print(f"Saved spectrograms for batch {batch_idx} to {self.output_dir}")
