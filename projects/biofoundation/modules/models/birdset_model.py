from typing import Literal, Optional
import datasets
from torch import nn
import torch
from birdset.configs import PretrainInfoConfig
from birdset.utils import pylogger
log = pylogger.get_pylogger(__name__)


class BioFoundationModel(nn.Module):
    def __init__(
        self,
        num_classes: int | None,
        embedding_size: int,
        classifier: nn.Module | None = None,
        local_checkpoint: str = None,
        load_classifier_checkpoint: bool = True,
        freeze_backbone: bool = False,
        preprocess_in_model: bool = False,
        pretrain_info: PretrainInfoConfig = None,
        pooling: Literal["default"] | None = "default",
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.local_checkpoint = local_checkpoint
        self.freeze_backbone = freeze_backbone
        self.preprocess_in_model = preprocess_in_model
        self.classifier = classifier
        self.embedding_size = embedding_size
        self.load_classifier_checkpoint = load_classifier_checkpoint
        self.pooling = pooling

        if pretrain_info:
            self.hf_path = pretrain_info.hf_path
            self.hf_name = (
                pretrain_info.hf_name
                if not pretrain_info.hf_pretrain_name
                else pretrain_info.hf_pretrain_name
            )
            if self.hf_path == "DBD-research-group/BirdSet":
                self.num_classes = len(
                    datasets.load_dataset_builder(self.hf_path, self.hf_name)
                    .info.features["ebird_code"]
                    .names
                )
            else:
                self.num_classes = num_classes
        else:
            self.hf_path = None
            self.hf_name = None
            self.num_classes = num_classes
        
        self.model = None
        self.preprocessor = None

        self.model = self._load_model()
        if self.preprocess_in_model:
            self.preprocessor = self._load_preprocessor()
        
        if freeze_backbone:
            self.freeze_model_backbone()

    def _load_preprocessor(self) -> nn.Module:
        # Implement this method in subclasses to load the preprocessor
        return None

    def _load_model(self) -> nn.Module:
        # Implement this method in subclasses to load the model
        raise NotImplementedError("Subclasses should implement this method to load the model.")
    
    def freeze_model_backbone(self):
        """
        Freezes the backbone of the model.
        """
        for param in self.model.parameters():
            param.requires_grad = False
        log.info(">> Backbone of the model is frozen.")

    def _preprocess(self, input_values: torch.Tensor) -> torch.Tensor:
        if self.preprocess_in_model:
            if self.preprocessor is None:
                raise ValueError("Preprocessor is not configured properly.")
            input_values = self.preprocessor(input_values)
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
        if self.preprocess_in_model:
            input_values = self._preprocess(input_values)
        if self.classifier is not None:
            embeddings = self.get_embeddings(input_values)
            logits = self.classifier(embeddings)
        else:
            output = self.model(input_values)
            logits = output.logits

        return logits
    
    def _load_local_checkpoint(self):
        state_dict = torch.load(self.local_checkpoint)["state_dict"]
        model_state_dict = {
            key.replace("model.model.", ""): weight
            for key, weight in state_dict.items() if key.startswith("model.model")
        }
        self.model.load_state_dict(model_state_dict)
        log.info(f">> Loaded model state dict from local checkpoint: {self.local_checkpoint}")


        # Process the keys for the classifier
        if self.classifier:
            if self.load_classifier_checkpoint:
                try:
                    classifier_state_dict = {
                        key.replace("model.classifier.", ""): weight
                        for key, weight in state_dict.items() if key.startswith("model.classifier.")
                    }
                    self.classifier.load_state_dict(classifier_state_dict)
                    log.info(f">> Also loaded classifier state dict from local checkpoint: {self.local_checkpoint}")
                except Exception as e:
                    log.error(f"Could not load classifier state dict from local checkpoint: {e}")  



    