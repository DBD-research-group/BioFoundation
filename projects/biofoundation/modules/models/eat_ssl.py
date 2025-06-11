from typing import Literal, Optional
from biofoundation.modules.models.vit import ViT
import torch
from torch import nn
import torch.nn.functional as F
import torchaudio


from transformers import AutoModel

from birdset.configs.model_configs import PretrainInfoConfig


class EATPreprocessor(nn.Module):
    MEAN = torch.tensor(-4.268)
    STD = torch.tensor(4.569)

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


class EATSSL(ViT):
    """
    Pretrained model for audio classification using the Efficient Audio Transformer (EAT) model.

    This file and the EAT folder includes code that is based on EAT by Wenxi Chen, licensed under the MIT License
    Copyright (c) 2024 Wenxi Chen
    Github-Repository: https://github.com/cwx-worst-one/EAT
    Paper: https://arxiv.org/abs/2401.03497

    We use a modified version of the EAT implementation that only relies on small local fairseq files and is compatible with Pytorch Lightning.
    This adaptation is by Paul Hahn and is also licensed under the MIT License.
    Github-Repository: https://github.com/nhaH-luaP/PyEat

    Important Parameters:
    ---------------------
    checkpoint: The path to the checkpoint to be loaded.
    multimodel: The settings for the Data2vec multimodel to be used in the model. This should best be defined in a hydra yaml.
    modality: The settings for the Image Encoder to be used in the model. This should best be defined in a hydra yaml.
    num_classes: Number of classification heads to be used in the model.
    train_classifier: If True, the model will output the embeddings and freeze the feature extractor. Default is False.
    """

    EMBEDDING_SIZE = 768
    MEAN = torch.tensor(-4.268)
    STD = torch.tensor(4.569)

    def __init__(
        self,
        num_classes: int | None,
        embedding_size: int = EMBEDDING_SIZE,
        checkpoint_path: str = "worstchan/EAT-base_epoch30_finetune_AS2M",
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

    def _load_model(self) -> None:
        """
        Load the model by using the Data2VecMultiModel and loading a local checkpoint. The decoder is not needed to extract features so we remove it and ignore its weights from the checkpoint.
        """
        return AutoModel.from_pretrained(self.checkpoint_path, trust_remote_code=True)

    def _load_preprocessor(self) -> nn.Module:
        """
        Load the preprocessor for the model.
        This is a Kaldi-like Mel spectrogram extractor.
        """
        return EATPreprocessor()

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

    def get_embeddings(self, input_tensor: torch.Tensor) -> torch.Tensor:
        features = self.model.extract_features(input_tensor)
        return self.pool(features, self.pooling_type)
