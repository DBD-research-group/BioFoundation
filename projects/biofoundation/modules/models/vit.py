from biofoundation.modules.models.birdset_model import BirdSetModel
from torch import nn
import torchvision.models as models
import torch
import torch.nn.functional as F
from torchaudio.compliance import kaldi
import torchaudio.transforms as T
from biofoundation.modules.models.inat_utils import (
    _MEL_BREAK_FREQUENCY_HERTZ,
    _MEL_HIGH_FREQUENCY_Q,
    AudioToImageConverter,
)


class ViTModel(BirdSetModel):
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
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            embedding_size=embedding_size,
            local_checkpoint=local_checkpoint,
            load_classifier_checkpoint=load_classifier_checkpoint,
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

        self.db_transform = T.AmplitudeToDB(
            top_db=80,
            stype="power",
        )

    def load_model(self) -> None:
        self.model = models.vit_b_16(weights=None)

        checkpoint_path = "/workspace/models/vit/vit_single_mixup.pt"
        state_dict = torch.load(checkpoint_path)

        keys_to_remove = [key for key in state_dict.keys() if "heads.head" in key]
        for key in keys_to_remove:
            del state_dict[key]

        self.model.load_state_dict(state_dict, strict=False)

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
        spectograms = self.preprocess(input_values)
        return self.model(spectograms)

    def preprocess(self, input_values: torch.Tensor) -> torch.Tensor:
        max_db_value = 0.0
        min_db_value = -100.0

        mel_db = self.mel_spectrogram(input_values)
        # mel_db = self.db_transform(mel_spec) breaks the training success

        mel_db_normalized = (mel_db - max_db_value) / (
            max_db_value - min_db_value + 1e-10
        )
        mel_img = (mel_db_normalized * 255).round().clamp(0, 255).to(torch.float32)
        mel_img = F.interpolate(
            mel_img, size=(224, 224), mode="bilinear", align_corners=False
        )
        mel_img = mel_img.repeat(1, 3, 1, 1)
        return mel_img
