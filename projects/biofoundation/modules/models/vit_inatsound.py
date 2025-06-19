# from biofoundation.modules.models.birdset_model import BirdSetModel
from typing import Literal, Optional, Tuple
from biofoundation.modules.models.vit import ViT
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as TV
import torchaudio.transforms as T
from torchaudio.compliance import kaldi
from birdset.configs.model_configs import PretrainInfoConfig
import timm
# from biofoundation.modules.models.biofoundation_model import BioFoundationModel


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
       
        '''self.mel_spectrogram = T.MelSpectrogram(
            window_fn=torch.hann_window,
            sample_rate=22050,
            n_fft=1024,
            win_length=256,
            hop_length=32,
            f_min=50,
            f_max=11025,
            n_mels=298,
            power=1.0,
        )'''
    
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=22050,
            n_fft=1024,
            win_length=512,
            hop_length=128,
            f_min=0,
            f_max=11025,
            n_mels=128,
            power=2.0,
            normalized=False,
            center=True,
            pad_mode="reflect",
            window_fn=torch.hann_window,
        )

        # self.model = None
        # self.load_model()

        #self._load_model()

    def _load_model(self) -> nn.Module:
        model = timm.create_model(
        'vit_base_patch16_224', 
        pretrained=True,  
        )
        #model = models.vit_b_16(weights=None)
        #checkpoint_path = "/workspace/models/vit/vit_single_mixup.pt"
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

    def preprocess_old(self, input_values: torch.Tensor) -> torch.Tensor:
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
    def preprocess(self, input_values: torch.Tensor) -> torch.Tensor:
        if input_values.ndim == 2:
            input_values = input_values.unsqueeze(1)  # (B, 1, T)

        mel_spec = self.mel_spectrogram(input_values)  # (B, 1, n_mels, T)
        mel_spec = mel_spec + 1e-10
        mel_spec = torch.log10(mel_spec)
        mel_spec = torch.clamp(mel_spec, min=-4.0, max=4.0)
        mel_spec = (mel_spec + 4.0) / 8.0  # normalize to [0,1]

        mel_spec = mel_spec.squeeze(1)  # (B, n_mels, T)
        mel_spec = mel_spec.unsqueeze(1)  # (B, 1, H, W)
        mel_spec = F.interpolate(mel_spec, size=(224, 224), mode="bilinear", align_corners=False)
        mel_spec = mel_spec.repeat(1, 3, 1, 1)  # (B, 3, 224, 224)
        return mel_spec


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
