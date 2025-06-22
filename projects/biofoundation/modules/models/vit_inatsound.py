# from biofoundation.modules.models.birdset_model import BirdSetModel
from typing import Literal, Optional, Tuple
from biofoundation.modules.models.vit import ViT
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
import torchaudio.transforms as T
from torchaudio.compliance import kaldi
from birdset.configs.model_configs import PretrainInfoConfig
import timm

# from biofoundation.modules.models.biofoundation_model import BioFoundationModel


class AudioToImageConverter(nn.Module):
    """GPU-accelerated audio to mel-spectrogram converter using PyTorch."""
    
    def __init__(
        self,
        samplerate: int = 22050,
        fft_length: int = 1024,
        window_length_samples: int = 512,
        hop_length_samples: int = 128,
        mel_bands: int = 128,
        mel_min_hz: float = 0,
        mel_max_hz: float = 11025,
        amin: float = 1e-10,
        ref_power_value: float = 1.0,
        max_db_value: float = 0.0,
        min_db_value: float = -100.0,
        **kwargs
    ):
        super().__init__()
        self.samplerate = samplerate
        self.fft_length = fft_length
        self.window_length_samples = window_length_samples
        self.hop_length_samples = hop_length_samples
        self.mel_bands = mel_bands
        self.mel_min_hz = mel_min_hz
        self.mel_max_hz = mel_max_hz
        self.amin = amin
        self.ref_power_value = ref_power_value
        self.max_db_value = max_db_value
        self.min_db_value = min_db_value
        
        # Key changes to match old implementation:
        # 1. Use power=2.0 then take sqrt to match magnitude calculation
        # 2. Use center=True to match scipy's default behavior
        # 3. Different mel scale to try to match librosa/scipy
        self.mel_transform = T.MelSpectrogram(
            sample_rate=samplerate,
            n_fft=fft_length,
            win_length=window_length_samples,
            hop_length=hop_length_samples,
            f_min=mel_min_hz,
            f_max=mel_max_hz,
            n_mels=mel_bands,
            power=2.0,  # Use power spectrum, then take sqrt
            normalized=False,
            center=True,  # Match scipy default
            pad_mode="reflect",  # Match scipy default
            window_fn=torch.hann_window,
            mel_scale="slaney",  # Try to match librosa default
        )
    
    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """
        Convert batch of waveforms to mel spectrograms with improved matching.
        """
        # Ensure correct input shape
        if waveforms.ndim == 1:
            waveforms = waveforms.unsqueeze(0)
        if waveforms.ndim == 2:
            waveforms = waveforms.unsqueeze(1)
        
        # Compute mel spectrogram (power spectrum)
        mel_spec = self.mel_transform(waveforms)
        mel_spec = mel_spec.squeeze(1)  # (B, n_mels, T)
        
        # Take square root to get magnitude from power
        mel_spec = torch.sqrt(mel_spec)
        
        # Apply scaling factor that matches scipy STFT scaling
        # The factor 2.0 matches the "* 2. / window.sum()" in old implementation
        window = torch.hann_window(self.window_length_samples, device=mel_spec.device)
        scale_factor = 2.0 / window.sum()
        mel_spec = mel_spec * scale_factor
        
        # Convert to dB - match preprocess.py exactly
        mel_spec = 20.0 * torch.log10(torch.clamp(mel_spec, min=self.amin))
        
        # Subtract reference power (dB conversion)
        mel_spec = mel_spec - 20.0 * torch.log10(torch.tensor(self.ref_power_value, device=mel_spec.device))
        
        # Clamp to dB range
        mel_spec = torch.clamp(mel_spec, min=self.min_db_value, max=self.max_db_value)
        
        # Normalize to [0, 255] range
        mel_spec = ((mel_spec - self.min_db_value) / (self.max_db_value - self.min_db_value)) * 255.0
        mel_spec = mel_spec.round().clamp(0, 255)
        
        # Apply orientation to match old implementation exactly
        # Transpose to [Time, Frequency]
        mel_spec = mel_spec.transpose(-2, -1)  # (B, T, F)
        
        # Flip frequency axis (high to low frequencies)
        mel_spec = torch.flip(mel_spec, dims=[-1])  # (B, T, F) with freq flipped
        
        # Transpose back to [Frequency, Time] for image format
        mel_spec = mel_spec.transpose(-2, -1)  # (B, F, T)
        
        return mel_spec


class Vit_iNatSoundModel(ViT):
    """
    ViT-B-16 model implemented like in the iNaturalist paper: https://openreview.net/pdf?id=QCY01LvyKm.
    """

    EMBEDDING_SIZE = 768

    def __init__(
        self,
        num_classes: int,
        embedding_size: int = EMBEDDING_SIZE,
        checkpoint_path: str = "/workspace/models/vit/vit_single_mixup.pt",
        local_checkpoint: str = None,
        load_classifier_checkpoint: bool = True,
        freeze_backbone: bool = False,
        preprocess_in_model: bool = True,
        classifier: nn.Module = None,
        pretrain_info: PretrainInfoConfig = None,
        pooling: Literal["just_cls", "attentive", "average"] = "just_cls",
    ) -> None:
        self.model = None  # Placeholder for the loaded model
        self.checkpoint_path = checkpoint_path
        super().__init__(
            num_classes=num_classes,
            embedding_size=embedding_size,
            local_checkpoint=local_checkpoint,
            load_classifier_checkpoint=load_classifier_checkpoint,
            freeze_backbone=freeze_backbone,
            preprocess_in_model=preprocess_in_model,
            pretrain_info=pretrain_info,
            pooling=pooling,
            classifier=classifier,
        )

        self.audio_to_image_converter = AudioToImageConverter(
            samplerate=22050,
            fft_length=1024,
            window_length_samples=512,
            hop_length_samples=128,
            mel_bands=128,
            mel_min_hz=0,
            mel_max_hz=11025,
            amin=1e-10,
            ref_power_value=1.,
            max_db_value=0.,
            min_db_value=-100.,
            target_height=None,
            target_width=None,
        )

        # self._load_model()

    def _load_model(self) -> nn.Module:
        model = timm.create_model(
            "vit_base_patch16_224",
            pretrained=True,
        )
        # model = models.vit_b_16(weights=None)
        # checkpoint_path = "/workspace/models/vit/vit_single_mixup.pt"
        state_dict = torch.load(self.checkpoint_path)

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
        """
        Process the input values with the audio to image converter and normalizer.
        Uses GPU batch processing for efficiency.

        Args:
            input_values (torch.Tensor): The input audio tensor of shape (B, T).

        Returns:
            torch.Tensor: The processed image tensor of shape (B, 3, 224, 224).
        """
        # Process entire batch at once on GPU
        mel_specs = self.audio_to_image_converter(input_values)  # (B, F, T)
        
        # Convert to 3-channel RGB format
        mel_specs = mel_specs.unsqueeze(1)  # (B, 1, F, T)
        mel_specs = mel_specs.repeat(1, 3, 1, 1)  # (B, 3, F, T)
        
        # Resize to 224x224 and apply normalization
        mel_specs = F.interpolate(mel_specs, size=(224, 224), mode='bilinear', align_corners=False)
        
        # Apply ImageNet normalization
        mel_specs = mel_specs / 255.0  # Convert from [0, 255] to [0, 1]
        mean = torch.tensor([0.6569, 0.6569, 0.6569], device=mel_specs.device).view(1, 3, 1, 1)
        std = torch.tensor([0.1786, 0.1786, 0.1786], device=mel_specs.device).view(1, 3, 1, 1)
        mel_specs = (mel_specs - mean) / std
        
        return mel_specs

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
        )  # shape (batch_size, 197, 768)
        return self.pool(embeddings, self.pooling_type)
