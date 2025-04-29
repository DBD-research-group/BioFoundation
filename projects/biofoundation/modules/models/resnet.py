import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torchaudio.compliance import kaldi
from biofoundation.modules.models.birdset_model import BirdSetModel


class ResNet50(BirdSetModel):
    def __init__(
        self,
        num_classes: int,
        embedding_size: int = 1000,
        local_checkpoint: str = None,
        load_classifier_checkpoint: bool = True,
        freeze_backbone: bool = False,
        preprocess_in_model: bool = True,
        classifier: nn.Module = None,
    ):
        super().__init__(
            num_classes=num_classes,
            embedding_size=embedding_size,
            local_checkpoint=local_checkpoint,
            load_classifier_checkpoint=load_classifier_checkpoint,
            preprocess_in_model=preprocess_in_model,
        )

        self.model = None
        self.load_model()

        if classifier is None:
            self.classifier = nn.Linear(embedding_size, num_classes)
        else:
            self.classifier = classifier

        if local_checkpoint:
            self._load_local_checkpoint()

    def load_model(self) -> None:
        self.model = models.resnet50(weights=None)

        checkpoint_path = "/workspace/models/resnet/r50_single_mixup.pt"
        state_dict = torch.load(checkpoint_path)
        state_dict.pop("fc.weight", None)
        state_dict.pop("fc.bias", None)

        self.model.load_state_dict(state_dict, strict=False)

    def forward(
        self, input_values: torch.Tensor, labels: torch.Tensor | None
    ) -> torch.Tensor:
        spectograms = self.preprocess(input_values)
        return self.classifier(self.model(spectograms))

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
