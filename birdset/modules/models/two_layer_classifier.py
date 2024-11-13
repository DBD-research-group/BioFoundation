import torch
from torch import nn
from typing import Optional

class TwoLayerClassifier(nn.Module):
    
    def __init__(
        self,
        num_classes: int,
        in_features: int,
        hidden_dim: int = 128,
        dropout_rate: float = 0.5
    ) -> None:
        """
        Initialize the LinearClassifier.
        
        Args:
            num_classes (int): The number of output classes for the classifier.
            in_features (int): The size of the input for the linear classifier.
        """
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes)
        )
         

    def forward(
        self, input_values: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self.classifier(input_values).squeeze(1)