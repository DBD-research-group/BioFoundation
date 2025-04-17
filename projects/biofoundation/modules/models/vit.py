from biofoundation.modules.models.birdset_model import BirdSetModel
from torch import nn
import torchvision.models as models
import torch
import torch.nn.functional as F
from torchaudio.compliance import kaldi


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

        if freeze_backbone:
            for name, param in self.model.named_parameters():
                if "heads.head" not in name:
                    param.requires_grad = False

    def load_model(self) -> None:
        self.model = models.vit_b_16(weights=None)

        checkpoint_path = "/workspace/models/vit/vit_single_mixup.pt"
        state_dict = torch.load(checkpoint_path)

        keys_to_remove = [key for key in state_dict.keys() if "heads.head" in key]
        for key in keys_to_remove:
            del state_dict[key]

        self.model.load_state_dict(state_dict, strict=False)

    def duplicate_channels(self, tensor):
        if tensor.shape[1] == 1:
            # Duplicate the single channel to three channels
            tensor = tensor.repeat(1, 3, 1, 1)
        return tensor

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

        device = input_values.device
        melspecs = []
        target_length = 512

        for waveform in input_values:
            if waveform.shape[-1] < 512:
                waveform = F.pad(waveform, (0, 512 - waveform.shape[-1]))
            melspec = kaldi.fbank(
                waveform,
                window_type="hanning",
                low_freq=50,
                high_freq=11025,
                sample_frequency=22050,
                num_mel_bins=128,
            )

            # Pad or crop mel spectrogram to fixed width (time dimension = 512)
            if melspec.shape[0] < target_length:
                pad_amount = target_length - melspec.shape[0]
                melspec = F.pad(melspec, (0, 0, 0, pad_amount))
            else:
                melspec = melspec[:target_length, :]

            # Normalize the dicibel converted mel spectrogram to the range [0, 255]
            epsilon = 1e-10  # Avoid log(0)
            melspec = 10 * torch.log10(melspec + epsilon)

            # Normalize to [0, 255] with fixed decibel range
            min_db = -80
            max_db = 0
            melspec = (melspec - min_db) / (max_db - min_db)
            melspec = torch.clamp(melspec, 0, 1) * 255

            # Convert to uint8
            melspec = melspec.to(torch.uint8).to(torch.float32)

            melspecs.append(melspec)

        melspecs = torch.stack(melspecs).to(device)
        melspecs = melspecs.unsqueeze(1)

        # Resize the tensor to [32, 1, 224, 224]
        melspecs = F.interpolate(
            melspecs, size=(224, 224), mode="bilinear", align_corners=False
        )

        # convert gray scale to 3 channels (RGB)
        melspecs = melspecs.repeat(1, 3, 1, 1)
        return melspecs
