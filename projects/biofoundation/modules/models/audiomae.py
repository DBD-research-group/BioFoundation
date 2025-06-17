from typing import Literal, Optional, Tuple
from biofoundation.modules.models.vit import ViT
import timm
import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
from torchaudio.compliance import kaldi

from birdset.configs.model_configs import PretrainInfoConfig


class KaldiLikeMelSpec(nn.Module):
    MEAN = -4.2677393
    STD = 4.5689974

    def __init__(self, target_frames: int = 1024):
        super().__init__()
        self.target_frames = target_frames
        self.preemphasis = torchaudio.functional.preemphasis
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=512,
            win_length=400,
            hop_length=160,
            window_fn=torch.hann_window,
            n_mels=128,
            f_min=20.0,
            f_max=8000.0,
            power=2.0,  # Kaldi default is power
            norm=None,
            mel_scale="htk",
        )
        # Use log-mel, not dB, for exact Kaldi parity

    def forward(self, x):
        # Remove DC offset
        x = x - x.mean(dim=-1, keepdim=True)
        # Pre-emphasis
        x = torchaudio.functional.preemphasis(x, coeff=0.97)
        melspecs = self.melspec(x)
        melspecs = torch.log(melspecs + 1e-6)
        n_frames = melspecs.shape[-1]
        if n_frames < self.target_frames:
            pad_amt = self.target_frames - n_frames
            melspecs = F.pad(melspecs, (0, pad_amt), mode="constant", value=0)
        else:
            melspec = melspec[..., : self.target_frames]
        melspecs = melspecs.permute(0, 1, 3, 2)  # (batch, 1, 128, 1024)
        melspecs = (melspecs - self.MEAN) / (self.STD * 2)
        return melspecs


class AudioMAEModel(ViT):
    """
    Pretrained model for audio classification using the AUDIOMAE model.
    Masked Autoencoders that Listen: https://arxiv.org/abs/2207.06405
    Pretrained weights from Huggingface: gaunernst/vit_base_patch16_1024_128.audiomae_as2m

    The model expect a 1D audio signale sampled with 16kHz and a length of 10s.
    """

    EMBEDDING_SIZE = 768
    MEAN = -4.2677393
    STD = 4.5689974

    def __init__(
        self,
        num_classes: int,
        embedding_size: int = EMBEDDING_SIZE,
        checkpoint_path: str = "hf_hub:gaunernst/vit_base_patch16_1024_128.audiomae_as2m",
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

        self.preprocessor = KaldiLikeMelSpec()

    def _load_model(self) -> None:
        """
        Load the model from Huggingface.
        """
        return timm.create_model(self.checkpoint_path, pretrained=True)

    def _load_preprocessor(self) -> nn.Module:
        """
        Load the preprocessor for the model.
        This is a Kaldi-like Mel spectrogram extractor.
        """
        return KaldiLikeMelSpec()

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
            logits = self.model(input_values)

        return logits

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

    def get_num_layers(self) -> int:
        """
        Get the number of layers in the model.

        Returns:
            int: The number of layers in the model.
        """
        return len(self.model.encoder.layers)
