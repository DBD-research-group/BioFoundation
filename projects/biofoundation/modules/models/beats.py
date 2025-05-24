from typing import Optional, Literal
from biofoundation.modules.models.AttentivePooling import AttentivePooling
import torch
from torch import nn


from biofoundation.modules.models.BEATs import BEATs, BEATsConfig
from biofoundation.modules.models.birdset_model import BirdSetModel

from birdset.utils import pylogger

log = pylogger.get_pylogger(__name__)


class BEATsModel(BirdSetModel):
    """
    Pretrained model for audio classification using the BEATs model.
    Expects a 1-channel 10s waveform input, all preprocessing is done in the network.
    """

    EMBEDDING_SIZE = 768

    def __init__(
        self,
        num_classes: int | None,
        embedding_size: int = EMBEDDING_SIZE,
        local_checkpoint: str = None,
        checkpoint_path: str = '/workspace/models/beats/BEATs_iter3_plus_AS2M.pt',
        load_classifier_checkpoint: bool = True,
        freeze_backbone: bool = False,
        preprocess_in_model: bool = True,
        classifier: nn.Module | None = None,
        pretrain_info = None,
        pooling: Literal['just_cls', 'attentive'] = "just_cls",
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            embedding_size=embedding_size,
            local_checkpoint=local_checkpoint,
            load_classifier_checkpoint=load_classifier_checkpoint,
            freeze_backbone=freeze_backbone,
            preprocess_in_model=preprocess_in_model,
            pretrain_info=pretrain_info,
        )
        self.model = None  # Placeholder for the loaded model
        self.checkpoint_path = checkpoint_path
        self.pooling = pooling
        if pooling == "attentive":
            attentive_heads = embedding_size // 8 # beats uses 8 heads
            self.attentive_pooling = AttentivePooling(
                embed_dim=embedding_size, num_heads=attentive_heads
            )
        self.load_model()
        if classifier is None:
            self.classifier = nn.Linear(embedding_size, num_classes)
        else:
            self.classifier = classifier

        if local_checkpoint:
            self._load_local_checkpoint()

        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

    def load_model(self) -> None:
        """
        Load the model from shared storage.
        """
        log.info(f">> Loading model from {self.checkpoint_path}")
        # load the pre-trained checkpoints
        checkpoint = torch.load(self.checkpoint_path)

        cfg = BEATsConfig(checkpoint["cfg"])
        self.model = BEATs(cfg)
        self.model.load_state_dict(checkpoint["model"])
        #self.model.predictor = None  # This should happen autom. if correct checkpoint
        self.model.eval()

    def _preprocess(self, input_values: torch.Tensor) -> torch.Tensor:
        """
        Preprocessing for the input values is done in BETAs.py
        The waveform gets resampled to 16kHz, transformed into a fbank and then normalized.
        """
        return input_values

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
        embeddings = self.get_embeddings(input_values, self.pooling)
        # flattend_embeddings = embeddings.reshape(embeddings.size(0), -1)
        if self.model.predictor is not None:
            return embeddings
        return self.classifier(embeddings)

    def get_embeddings(self, input_values: torch.Tensor, pooling) -> torch.Tensor:
        """
        Get the embeddings and logits from the BEATs model.

        Args:
            input_tensor (torch.Tensor): The input tensor for the model.

        Returns:
            torch.Tensor: The embeddings from the model.
        """
        if self.preprocess_in_model:
            input_values = self._preprocess(input_values)
        embeddings = self.model.extract_features(input_values)[0]
        if pooling == "just_cls" and self.model.predictor is None: # It returns Probabilities otherwise
            # Use only the CLS token for classification
            # The CLS token is the first token in the sequence
            return embeddings[:, 0, :]
        elif pooling == "attentive":
            return self.attentive_pooling(embeddings)

        return embeddings
