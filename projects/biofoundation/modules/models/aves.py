import torch
import torch.nn as nn
from torchaudio.models import wav2vec2_model
import json
from typing import Optional, Literal

from birdset.configs import PretrainInfoConfig

from biofoundation.modules.models.vit import ViT


class AvesClassifier(ViT):
    """
    Pretrained model for audio classification using the AVES model.

    This file includes code from AVES by Masato Hagiwara, licensed under the MIT License
    Copyright (c) 2022 Earth Species Project
    Github-Repository: https://github.com/earthspecies/aves
    Paper: https://arxiv.org/abs/2210.14493
    """

    EMBEDDING_SIZE = 768

    def __init__(
        self,
        num_classes: int = None,
        embedding_size: int = EMBEDDING_SIZE,
        checkpoint: str = "/workspace/models/aves/aves-base-bio.torchaudio.pt",
        config: str = "/workspace/models/aves/aves-base-bio.torchaudio.model_config.json",
        local_checkpoint: str = None,
        load_classifier_checkpoint: bool = True,
        freeze_backbone: bool = False,
        preprocess_in_model: bool = True,
        classifier: nn.Module | None = None,
        pretrain_info: PretrainInfoConfig = None,
        pooling: Literal["just_cls", "attentive", "average"] = "just_cls",
    ):
        self.model = None  # Placeholder for the loaded model
        self.checkpoint_path = checkpoint
        self.config_path = config
        super().__init__(
            num_classes=num_classes,
            embedding_size=embedding_size,
            local_checkpoint=local_checkpoint,
            load_classifier_checkpoint=load_classifier_checkpoint,
            freeze_backbone=freeze_backbone,
            preprocess_in_model=preprocess_in_model,
            pretrain_info=pretrain_info,
        )
        if classifier is None:
            self.classifier = nn.Linear(embedding_size, num_classes)
        else:
            self.classifier = classifier

    def _load_model(self) -> None:
        """
        Load the model from shared storage.
        """
        self.config = self.load_config(self.config_path)
        model = wav2vec2_model(**self.config, aux_num_out=None)
        model.load_state_dict(torch.load(self.checkpoint_path))
        model.feature_extractor.requires_grad_(True)
        return model

    def load_config(self, config_path):
        with open(config_path, "r") as ff:
            obj = json.load(ff)

        return obj

    def _load_preprocessor(self) -> nn.Module:
        """
        Load the preprocessor for the model.
        This is a Kaldi-like Mel spectrogram extractor.
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
            logits = self.model(input_values)

        return logits

    def get_embeddings(self, input_values: torch.Tensor) -> torch.Tensor:
        """
        Get the embeddings and logits from the BEATs model.

        Args:
            input_tensor (torch.Tensor): The input tensor for the model.

        Returns:
            torch.Tensor: The embeddings from the model.
        """

        input_values = input_values.squeeze(1)
        features = self.model.extract_features(input_values)[0][-1]

        return self.pool(features, self.pooling_type)
