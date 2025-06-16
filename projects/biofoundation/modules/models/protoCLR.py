from typing import Literal, Optional, Tuple
from biofoundation.modules.models.Pooling import AttentivePooling, AveragePooling
from biofoundation.modules.models.ProtoCLR.cvt import cvt13
import torch
from torch import nn
import torch.nn.functional as F


from biofoundation.modules.models.biofoundation_model import BioFoundationModel

from birdset.configs.model_configs import PretrainInfoConfig

from biofoundation.modules.models.ProtoCLR.melspectrogram import MelSpectrogramProcessor


class ProtoCLRModel(BioFoundationModel):
    """
    Pretrained model for bird classification using Domain-Invariant Representation Learning of Bird Sounds

    The code in this file is based / copied from ProtoCLR by Ilyass Moummad et al.
    Github-Repository: https://github.com/ilyassmoummad/ProtoCLR
    Paper: https://arxiv.org/abs/2409.08589
    """

    EMBEDDING_SIZE = 384

    def __init__(
        self,
        num_classes: int | None,
        embedding_size: int = EMBEDDING_SIZE,
        checkpoint_path: str = "/workspace/models/protoclr/protoclr.pth",
        local_checkpoint: str = None,
        load_classifier_checkpoint: bool = True,
        freeze_backbone: bool = False,
        preprocess_in_model: bool = True,
        classifier: nn.Module | None = None,
        pretrain_info: PretrainInfoConfig = None,
        pooling: Literal["just_cls", "attentive", "average"] = "just_cls",
    ) -> None:
        self.model = None
        self.checkpoint_path = checkpoint_path
        super().__init__(
            num_classes=num_classes,
            embedding_size=embedding_size,
            local_checkpoint=local_checkpoint,
            load_classifier_checkpoint=load_classifier_checkpoint,
            freeze_backbone=freeze_backbone,
            preprocess_in_model=preprocess_in_model,
            pretrain_info=pretrain_info,
            pooling=pooling,
            classifier=classifier,
        )

        if self.pooling == "default":
            self.pooler = nn.LayerNorm(
                self.config.hidden_sizes[-1], eps=self.config.layer_norm_eps
            )
        elif self.pooling == "attentive":
            self.pooler = AttentivePooling(dim=embedding_size, num_heads=8)
        elif self.pooling == "average":
            self.pooler = AveragePooling()

    def _load_model(self) -> None:
        model = cvt13()
        model.load_state_dict(torch.load(self.checkpoint_path, map_location="cpu"))

        return model

    def _load_preprocessor(self):
        return MelSpectrogramProcessor()

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
        embeddings = self.get_embeddings(input_values)

        return self.classifier(embeddings)

    def get_embeddings(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Get the embeddings and logits from the AUDIOMAE model.

        Args:
            input_tensor (torch.Tensor): The input tensor for the model.

        Returns:
            torch.Tensor: The embeddings from the model.
        """
        if self.preprocess_in_model:
            input_values = self._preprocess(input_tensor)

        output, cls_tokens = self.model.output_embeddings(input_values)
        if self.pooling == "just_cls":
            embeddings = cls_tokens
        elif self.pooling == "attentive" or self.pooling == "average":
            embeddings = self.pooler(output)

        return embeddings
