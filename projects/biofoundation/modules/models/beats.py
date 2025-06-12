from typing import Optional, Literal
from biofoundation.modules.models.vit import ViT

import torch
from torch import nn


from biofoundation.modules.models.BEATs import BEATs, BEATsConfig

from birdset.utils import pylogger

log = pylogger.get_pylogger(__name__)


class BEATsModel(ViT):
    """
    Pretrained model for audio classification using the BEATs model.
    Expects a 1-channel 10s waveform input, all preprocessing is done in the network.
    """

    EMBEDDING_SIZE = 768

    def __init__(
        self,
        num_classes: int | None,
        embedding_size: int = EMBEDDING_SIZE,
        checkpoint_path: str = "/workspace/models/beats/BEATs_iter3_plus_AS2M.pt",
        local_checkpoint: str = None,
        load_classifier_checkpoint: bool = True,
        freeze_backbone: bool = False,
        preprocess_in_model: bool = True,
        classifier: nn.Module | None = None,
        pretrain_info=None,
        pooling: Literal[
            "just_cls", "attentive", "attentive_old", "average", "mean"
        ] = "just_cls",
    ) -> None:
        self.model = None  # Placeholder for the loaded model
        self.checkpoint_path = checkpoint_path
        super().__init__(
            num_classes=num_classes,
            embedding_size=embedding_size,
            classifier=classifier,
            local_checkpoint=local_checkpoint,
            load_classifier_checkpoint=load_classifier_checkpoint,
            freeze_backbone=freeze_backbone,
            preprocess_in_model=preprocess_in_model,
            pretrain_info=pretrain_info,
            pooling=pooling,
        )
        

    def _load_model(self) -> None:
        """
        Load the model from shared storage.
        """
        log.info(f">> Loading model from {self.checkpoint_path}")
        # load the pre-trained checkpoints
        checkpoint = torch.load(self.checkpoint_path)

        cfg = BEATsConfig(checkpoint["cfg"])
        model = BEATs(cfg)
        model.load_state_dict(checkpoint["model"])
        # self.model.predictor = None  # This should happen autom. if correct checkpoint
        return model

    def _load_preprocessor(self):
        """
        Preprocessing for the input values is done in BETAs.py
        The waveform gets resampled to 16kHz, transformed into a fbank and then normalized.
        """
        return nn.Identity()
    
    def forward(
        self, input_values: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            input_values (torch.Tensor): The input tensor for the classifier.
            labels (Optional[torch.Tensor]): The true labels for the input values. Default is None.

        Returns:
            torch.Tensor: The output of the classifier.
        """
        if self.preprocess_in_model:
            input_values = self._preprocess(input_values)
        if self.classifier is not None:
            embeddings = self.get_embeddings(input_values)
            logits = self.classifier(embeddings)
        else:
           logits = self.model(input_values)[0]

        return logits

    def get_embeddings(self, input_values: torch.Tensor) -> torch.Tensor:
        """
        Get the embeddings and logits from the BEATs model.

        Args:
            input_tensor (torch.Tensor): The input tensor for the model.

        Returns:
            torch.Tensor: The embeddings from the model.
        """
        embeddings = self.model.extract_features(input_values)[
            0
        ]  # outputs a tensor of size batch_sizex496x768
        return self.pool(embeddings, self.pooling_type)

    def get_num_layers(self) -> int:
        """
        Get the number of layers in the model.

        Returns:
            int: The number of layers in the model.
        """
        return len(self.model.encoder.layers)
