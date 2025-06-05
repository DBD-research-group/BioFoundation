from typing import Optional
from biofoundation.modules.models.vit import ViT
import torch
from torch import nn
import torch.nn.functional as F
from torchaudio.compliance import kaldi


from transformers import AutoModel



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
        classifier: nn.Module | None = None,
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
        )

    def _load_model(self) -> None:
        """
        Load the model by using the Data2VecMultiModel and loading a local checkpoint. The decoder is not needed to extract features so we remove it and ignore its weights from the checkpoint.
        """
        return AutoModel.from_pretrained(self.checkpoint_path, trust_remote_code=True)


    def _preprocess(self, input_values: torch.Tensor) -> torch.Tensor:
        """
        Preprocesses the input values by applying mel-filterbank transformation. Similar as function for AudioMae, ConvNeXt and SSAST.
        Args:
            input_values (torch.Tensor): Input tensor of shape (batch_size, num_samples).
        Returns:
            torch.Tensor: Preprocessed tensor of shape (batch_size, 1, num_mel_bins, num_frames).
        """
        device = input_values.device
        melspecs = []
        for waveform in input_values:
            waveform = waveform - waveform.mean()
            melspec = kaldi.fbank(
                waveform,
                htk_compat=True,
                sample_frequency=16000,
                use_energy=False,
                window_type='hanning',
                num_mel_bins=128,
                dither=0.0,
                frame_shift=10
            ).unsqueeze(0)
            # Pad or truncate
            n_frames = melspec.shape[1]
            if n_frames < 1024:
                melspec = torch.nn.ZeroPad2d((0, 0, 0, 1024 - n_frames))(melspec)
            else:
                melspec = melspec[:, :1024, :]
            melspecs.append(melspec)
        melspecs = torch.stack(melspecs).to(device)

        melspecs = (melspecs - self.MEAN) / (self.STD * 2)
        return melspecs

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
