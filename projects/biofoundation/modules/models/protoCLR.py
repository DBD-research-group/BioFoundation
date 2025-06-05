from typing import Optional, Tuple
from biofoundation.modules.models.ProtoCLR.cvt import cvt13
import torch
from torch import nn
import torch.nn.functional as F


from biofoundation.modules.models.birdset_model import BioFoundationModel

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
        classifier: nn.Module | None = None,
        local_checkpoint: str = None,
        load_classifier_checkpoint: bool = True,
        freeze_backbone: bool = False,
        preprocess_in_model: bool = True,
        pretrain_info: PretrainInfoConfig = None,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            embedding_size=embedding_size,
            local_checkpoint=local_checkpoint,
            load_classifier_checkpoint=load_classifier_checkpoint,
            freeze_backbone=freeze_backbone,
            preprocess_in_model=preprocess_in_model,
        )
        self.model = None  # Placeholder for the loaded model
        self.preprocessor = None  # Placeholder for the preprocessor
        self.load_model()

        if preprocess_in_model:
            self.preprocessor = MelSpectrogramProcessor()


        # Define a linear classifier to use on top of the embeddings
        if classifier is None:
            self.classifier = nn.Linear(embedding_size, num_classes)
        else:
            self.classifier = classifier

        if local_checkpoint:
            self._load_local_checkpoint()
            
        # freeze the model
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

    def load_model(self) -> None:
       self.model = cvt13()
       self.model.load_state_dict(torch.load("/workspace/models/protoclr/protoclr.pth", map_location="cpu"))


    def preprocess(self, input_values: torch.Tensor) -> torch.Tensor:
        return self.preprocessor(input_values)

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
            input_values = self.preprocess(input_tensor)

        output = self.model.forward_features(input_values)
        return output