from biofoundation.modules.models.birdset_model import BirdSetModel
from torch import nn
import torchvision.models as models
import torch


class ViTModel(BirdSetModel):
    EMBEDDING_SIZE = 768

    def __init__(
        self,
        num_classes: int,
        embedding_size: int = EMBEDDING_SIZE,
        local_checkpoint: str = None,
        load_classifier_checkpoint: bool = True,
        freeze_backbone: bool = False,
        preprocess_in_model: bool = True,
        classifier: nn.Module = None,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            embedding_size=embedding_size,
            local_checkpoint=local_checkpoint,
            load_classifier_checkpoint=load_classifier_checkpoint,
            freeze_backbone=freeze_backbone,
            preprocess_in_model=preprocess_in_model,
        )

        self.num_classes = num_classes
        self.model = None
        self.load_model()

        if classifier is None:
            self.model.heads.head = nn.Linear(embedding_size, num_classes)
        else:
            self.model.heads.head = classifier

        if local_checkpoint:
            self._load_local_checkpoint()

        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

    def load_model(self) -> None:
        self.model = models.vit_b_16(weights=None)

        checkpoint_path = "/workspace/models/vit/vit_single_mixup.pt"
        state_dict = torch.load(checkpoint_path)

        self.model.load_state_dict(state_dict)
        self.model.eval()

    def preprocess(self, input_values: torch.Tensor) -> torch.Tensor:
        # needed?
        pass

    def forward(
        self, input_values: torch.Tensor, labels: torch.Tensor | None
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
