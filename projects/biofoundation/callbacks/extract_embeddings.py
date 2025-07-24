import os
import torch
import numpy as np
from lightning.pytorch.callbacks import Callback
from biofoundation.modules.models.vit import ViT


class ExtractEmbeddings(Callback):
    def __init__(
        self,
        output_dir: str,
        dataset: str,
    ):
        super().__init__()
        self.dataset = dataset
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = os.path.join(output_dir, self.dataset)
        os.makedirs(self.output_dir, exist_ok=True)
        self.embeddings = []
        self.labels = []

    def on_test_epoch_end(self, trainer, pl_module):
        model = pl_module.model
        model_name = model.__class__.__name__
        # If model is a subclass of ViT, append pooling type to name
        if isinstance(model, ViT) and hasattr(model, "pooling_type"):
            model_name = f"{model_name}_{model.pooling_type}"
        embeddings_path = os.path.join(self.output_dir, f"{model_name}.npy")
        labels_path = os.path.join(self.output_dir, f"labels_{model_name}.npy")
        np.save(embeddings_path, np.array(self.embeddings))
        np.save(labels_path, np.array(self.labels))
        print(f"Saved embeddings to {embeddings_path}")
        print(f"Saved labels to {labels_path}")

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        model = pl_module.model
        inputs = batch["input_values"]
        # Check if _preprocess exists as a function before calling
        if model.preprocess_in_model and model.__class__.__name__ != "PerchModel":
            inputs = model._preprocess(inputs)
        embeddings = model.get_embeddings(inputs)
        labels = batch.get("labels", None)
        for i in range(embeddings.size(0)):
            self.embeddings.append(embeddings[i].detach().cpu())
            if labels is not None:
                self.labels.append(labels[i].detach().cpu())
