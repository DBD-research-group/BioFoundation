from email.mime import audio
import torch
import torch.nn as nn
from transformers import ClapModel, ClapProcessor

from birdset.configs import PretrainInfoConfig
from typing import Optional, Literal
from transformers import pipeline
from biofoundation.modules.models.Pooling import (
    AttentivePooling,
    AttentivePooling_old,
    AveragePooling,
)
from birdset.utils import pylogger
from biofoundation.modules.models.biofoundation_model import BioFoundationModel

log = pylogger.get_pylogger(__name__)


class BioLingualClassifier(BioFoundationModel):
    """
    Pretrained model for audio classification using the Biolingual model.

    This file includes code from BioLingual by David Robinson, licensed under the Apache-2.0 License
    Github-Repository: https://github.com/david-rx/BioLingual
    Paper: https://arxiv.org/abs/2308.04978

    Important Parameters:
    ---------------------
    checkpoint: Path to the AVES model checkpoint.
    n_last_hidden_layer: Number of last hidden layer (from the back) to extract the embeddings from. Default is 1.
    train_classifier: If True, the model will output the embeddings and freeze the feature extractor. Default is False.
    """

    EMBEDDING_SIZE = 768

    def __init__(
        self,
        num_classes: int = None,
        embedding_size: int = EMBEDDING_SIZE,
        checkpoint: str = "davidrrobinson/BioLingual",
        local_checkpoint: str = None,
        load_classifier_checkpoint: bool = True,
        freeze_backbone: bool = False,
        preprocess_in_model: bool = True,
        classifier: nn.Module = None,
        pretrain_info: PretrainInfoConfig = None,
        device: int | str = "cuda",
        pooling: Literal[
            "just_cls", "attentive", "attentive_old", "average", "mean"
        ] = "just_cls",
    ):
        """
        Note: Either num_classes or pretrain_info must be given
        Args:
            checkpoint: huggingface checkpoint path of any model of correct type
            num_classes: number of classification heads to be used in the model
            local_checkpoint: local path to checkpoint file
            cache_dir: specified cache dir to save model files at
            pretrain_info: hf_path and hf_name of info will be used to infer if num_classes is None
        """
        self.checkpoint_path = checkpoint
        self.device = device
        super().__init__(
            num_classes=num_classes,
            embedding_size=embedding_size,
            classifier=classifier,
            local_checkpoint=local_checkpoint,
            load_classifier_checkpoint=load_classifier_checkpoint,
            freeze_backbone=freeze_backbone,
            preprocess_in_model=preprocess_in_model,
            pretrain_info=pretrain_info,
        )
        self.pooling_type = pooling
        if self.pooling_type == "attentive":
            attentive_heads = 32
            self.attentive_pooling = AttentivePooling(
                dim=embedding_size, num_heads=attentive_heads
            )
        elif self.pooling_type == "attentive_old":
            attentive_heads = 32
            self.attentive_pooling = AttentivePooling_old(
                embed_dim=embedding_size, num_heads=attentive_heads
            )
        elif self.pooling_type == "average":
            self.average_pooling = AveragePooling()

    def _load_model(self) -> None:
        """
        Load the model from shared storage.
        """
        log.info(f">> Loading model from {self.checkpoint_path}")
        model = ClapModel.from_pretrained(self.checkpoint_path).to(self.device)

        if self.preprocess_in_model:
            self.processor = ClapProcessor.from_pretrained(
                self.checkpoint_path
            )  # This takes too much memory if loaded in addition to one in transforms
        return model

    def _preprocess(self, input_values: torch.Tensor) -> torch.Tensor:
        """
        Preprocessing for the input values is done in BETAs.py
        The waveform gets resampled to 16kHz, transformed into a fbank and then normalized.
        """
        if self.preprocess_in_model:
            input_values = input_values.squeeze(1)
            inputs = self.processor(
                audios=input_values.cpu().numpy(),
                return_tensors="pt",
                sampling_rate=48000,
            )  # .input_features.to(input_values.device)
            inputs["input_features"] = inputs["input_features"].to(self.device)
            return inputs

        else:

            return {'input_features': input_values, 'is_longer': False}

    def pool(self, embeddings: torch.Tensor, pooling) -> torch.Tensor:
        if pooling == "just_cls":
            # Use only the CLS token for classification
            # The CLS token is the first token in the sequence
            return embeddings[:, 0, :]
        elif pooling == "attentive":
            embeddings = embeddings[:, 1:, :]
            return self.attentive_pooling(embeddings)
        elif pooling == "attentive_old":
            embeddings = embeddings[:, 1:, :]
            return self.attentive_pooling(embeddings)
        elif pooling == "average":
            return self.average_pooling(embeddings)
        else:
            raise ValueError(
                f"Pooling method '{pooling}' is not supported. Choose from 'just_cls', 'attentive', 'attentive_old', or 'average'."
            )

    def get_num_layers(self) -> int:
        """
        Returns the number of layers in the model.
        """
        return len(self.model.encoder.layers)

    def forward(
        self, input_values: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        embeddings = self.get_embeddings(input_values)

        return self.classifier(embeddings)

    def get_embeddings(self, input_tensor) -> torch.Tensor:
        inputs = self._preprocess(input_tensor)
       
        output = self.model.audio_model(
            **inputs, output_hidden_states=True, return_dict=True
        )  # self.model.extract_audio_features() doesnt retrn the hidden states for some reason so we go a layer down
        embeddings = output.last_hidden_state
        B, D, H, W = embeddings.shape

        # Flatten the 2D patch grid into a sequence
        embeddings = embeddings.permute(0, 2, 3, 1).reshape(B, H * W, D)

        return self.pool(embeddings, self.pooling_type)
