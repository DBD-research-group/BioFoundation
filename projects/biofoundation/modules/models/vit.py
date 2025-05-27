from biofoundation.modules.models.birdset_model import BirdSetModel
from torch import nn
import torchvision.models as models
import torch
import torch.nn.functional as F
from torchaudio.compliance import kaldi
import torchaudio.transforms as T
from biofoundation.modules.models.inat_utils import AudioToImageConverter

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
            win_length=512,
            hop_length=128,
            f_min=50,
            f_max=11025,
            n_mels=128,
            power=2.0,
        )

        self.db_transform = T.AmplitudeToDB(
            top_db=80,
            stype="power",
        )

        self.converter = AudioToImageConverter(
                        freq_scale='mel',
                        samplerate=22050,
                        fft_length=1024,
                        window_length_samples=256,
                        hop_length_samples=32,
                        mel_bands=128,
                        mel_min_hz=50,
                        mel_max_hz=8000,
                        max_db_value=0.,
                        min_db_value=-80.,
                        target_height=224,
                        target_width=224,
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
        spectograms = self.preprocess4(input_values)
        return self.model(spectograms)

    def preprocess4(self, input_values):
        input_values = input_values.detach().cpu().numpy()
        spec = self.converter._samples_to_magnitude_spectrogram(input_values)

        if self.converter.freq_scale == 'mel':
            spec = self.converter._mel_scale(spec)

        spec = self.converter._magnitude_to_db(spec)
        spec = self.converter._db_to_uint8(spec)
        spec = self.converter._orientate(spec)
        spec, _, _ = self.converter._resize(spec)

        spec_tensor = torch.tensor(spec, dtype=torch.float32, device="cuda")

        return spec_tensor


    def preprocess3(self, input_values: torch.Tensor) -> torch.Tensor:
        mel_db = self.mel_spectrogram(input_values)
        # mel_db = self.db_transform(mel_spec) breaks the training success

        mel_db_normalized = (mel_db - mel_db.min()) / (
            mel_db.max() - mel_db.min() + 1e-10
        )
        mel_img = (mel_db_normalized * 255).round().clamp(0, 255).to(torch.float32)
        mel_img = F.interpolate(
            mel_img, size=(224, 224), mode="bilinear", align_corners=False
        )
        mel_img = mel_img.repeat(1, 3, 1, 1)
        return mel_img

    def preprocess2(self, input_values: torch.Tensor) -> torch.Tensor:
        input_values = F.interpolate(
            input_values, size=(224, 224), mode="bilinear", align_corners=False
        )
        input_values = input_values.repeat(1, 3, 1, 1)
        return input_values

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
