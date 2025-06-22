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

from biofoundation.modules.models.ViT_INS.preprocess import AudioToImageConverter

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

        self.audio_to_image_converter = AudioToImageConverter(
            freq_scale='mel',
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
            save_distribution=None
        )

        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(
                (0.6569, 0.6569, 0.6569), (0.1786, 0.1786, 0.1786)
            ),
        ])

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

        Args:
            input_values (torch.Tensor): The input audio tensor.

        Returns:
            torch.Tensor: The processed image tensor.
        """
        device = input_values.device
        input_values = input_values.to("cpu")  # Move to CPU for processing
        preprocessed_images = []
        for waveform in input_values:
            mel_spec = self.audio_to_image_converter(waveform)
            mel_spec = mel_spec.copy()
            mel_spec = np.stack([mel_spec]*3) # Convert to 3 channels
            mel_spec = torch.from_numpy(mel_spec).float()

            preprocessed_images.append(mel_spec)
        # Stack the preprocessed images into a single tensor

        mel_specs = torch.stack(preprocessed_images, dim=0)
        mel_specs = mel_specs.to(device)
        mel_specs = self.transforms(mel_specs)

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
