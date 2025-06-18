# from biofoundation.modules.models.birdset_model import BirdSetModel
from torch import nn
import torchvision.models as models
import torch
import torch.nn.functional as F
from torchaudio.compliance import kaldi
import torchaudio.transforms as T
import torchvision.transforms as TV

# from biofoundation.modules.models.biofoundation_model import BioFoundationModel
from biofoundation.modules.models.vit import ViT
from birdset.configs.model_configs import PretrainInfoConfig
from typing import Literal, Optional, Tuple


class Vit_iNatSoundModel(ViT):
    """
    ViT-B-16 model implemented like in the iNaturalist paper: https://openreview.net/pdf?id=QCY01LvyKm.
    """

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
        pretrain_info: PretrainInfoConfig = None,
        pooling: Literal["just_cls", "attentive", "average"] = "just_cls",
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            embedding_size=embedding_size,
            local_checkpoint=local_checkpoint,
            load_classifier_checkpoint=load_classifier_checkpoint,
            preprocess_in_model=preprocess_in_model,
            pooling=pooling,
        )

        self.mel_spectrogram = T.MelSpectrogram(
            window_fn=torch.hann_window,
            sample_rate=22050,
            n_fft=1024,
            win_length=256,
            hop_length=32,
            f_min=50,
            f_max=11025,
            n_mels=298,
            power=1.0,
        )

        self.num_classes = num_classes
        # self.model = None
        # self.load_model()

        if classifier is None:
            self.model.heads.head = nn.Linear(embedding_size, num_classes)
        else:
            self.model.heads.head = classifier

        if local_checkpoint:
            self._load_local_checkpoint()

        if freeze_backbone:
            for name, param in self.model.named_parameters():
                if "heads.head" not in name:
                    param.requires_grad = False

    def _load_model(self) -> nn.Module:
        model = models.vit_b_16(weights=None)

        checkpoint_path = "/workspace/models/vit/vit_single_mixup.pt"
        state_dict = torch.load(checkpoint_path)

        keys_to_remove = [key for key in state_dict.keys() if "heads.head" in key]
        for key in keys_to_remove:
            del state_dict[key]

        model.load_state_dict(state_dict, strict=False)
        return model

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
        # input_values = self.duplicate_channels(input_values)
        # spectograms = self.preprocess(input_values)
        # return self.model(spectograms)
        if self.preprocess_in_model:
            input_values = self.preprocess(input_values)
        if self.classifier is not None:
            embeddings = self.get_embeddings(input_values)
            logits = self.classifier(embeddings)
        else:
            logits = self.model(input_values)

        return logits

    def preprocess(self, input_values: torch.Tensor) -> torch.Tensor:
        max_db_value = 0.0
        min_db_value = -100.0

        mel_db = self.mel_spectrogram(input_values)

        mel_db_normalized = (mel_db - max_db_value) / (
            max_db_value - min_db_value + 1e-10
        )
        mel_img = (mel_db_normalized * 255).round().clamp(0, 255).to(torch.float32)
        mel_img = F.interpolate(
            mel_img, size=(224, 224), mode="bilinear", align_corners=False
        )
        mel_img = mel_img.repeat(1, 3, 1, 1)
        return mel_img

    def get_embeddings(self, input_values: torch.Tensor) -> torch.Tensor:
        """
        Get the embeddings and logits from the AUDIOMAE model.

        Args:
            input_tensor (torch.Tensor): The input tensor for the model.

        Returns:
            torch.Tensor: The embeddings from the model.
        """
        embeddings = self.model.forward_features(
            input_values
        )  # shape (batch_size, 513, 768)
        return self.pool(embeddings, self.pooling_type)
