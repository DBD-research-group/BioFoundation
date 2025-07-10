from typing import Dict, Literal, Optional

from biofoundation.modules.models.Pooling import AttentivePooling, AveragePooling
import torch
from torch import nn
import torchaudio
from transformers import AutoConfig
from transformers.models.convnext.modeling_convnext import ConvNextModel
from birdset.configs import PretrainInfoConfig
from typing import Tuple

from transformers import AutoConfig, ConvNextForImageClassification

from torchvision import transforms

from biofoundation.modules.models.biofoundation_model import BioFoundationModel
from birdset.modules.models.convnext import ConvNextClassifier

from birdset.datamodule.components.augmentations import PowerToDB


class ConvNextModule(BioFoundationModel):
    """
    ConvNext model for audio classification.
    """

    EMBEDDING_SIZE = 1024

    def __init__(
        self,
        num_classes: Optional[int] = None,
        embedding_size: int = EMBEDDING_SIZE,
        freeze_backbone: bool = False,
        preprocess_in_model: bool = False,
        classifier: nn.Module | None = None,
        restrict_logits: bool = False,
        num_channels: int = 1,
        checkpoint: str = "DBD-research-group/ConvNeXT-Base-BirdSet-XCL",
        local_checkpoint: Optional[str] = None,
        load_classifier_checkpoint: bool = True,
        cache_dir: Optional[str] = None,
        pretrain_info: PretrainInfoConfig = None,
        pooling: Literal["default", "average", "attentive"] = "default",
    ):
        """
        Initializes the ConvNeXT-based model with configurable options for classification, checkpoint loading, and pooling.
            num_classes (Optional[int], optional): Number of output classes for classification. If None, must provide `pretrain_info`. Defaults to None.
            embedding_size (int, optional): Size of the embedding layer. Defaults to EMBEDDING_SIZE.
            freeze_backbone (bool, optional): If True, freezes the backbone during training. Defaults to False.
            preprocess_in_model (bool, optional): If True, applies preprocessing within the model. Defaults to False.
            classifier (Optional[nn.Module], optional): Custom classifier head. If None, a default classifier is used. Defaults to None.
            num_channels (int, optional): Number of input channels for the model. Defaults to 1.
            checkpoint (str, optional): HuggingFace checkpoint path for loading pretrained weights. Defaults to 'DBD-research-group/ConvNeXT-Base-BirdSet-XCL'.
            local_checkpoint (Optional[str], optional): Local path to a checkpoint file. If provided, loads weights from this file. Defaults to None.
            load_classifier_checkpoint (bool, optional): Whether to load the classifier weights from the checkpoint. Defaults to True.
            cache_dir (Optional[str], optional): Directory to cache model files. Defaults to None.
            pretrain_info (PretrainInfoConfig, optional): Configuration for pretraining information. Used to infer `num_classes` if not provided. Defaults to None.
            pooling (Literal["default", "average", "attentive"], optional): Pooling strategy to use in the model. Defaults to "default".
        Notes:
            - Either `num_classes` or `pretrain_info` must be provided.
            - The model supports loading from both HuggingFace and local checkpoints.
            - Pooling can be set to "default" (LayerNorm), "average", or "attentive" (AttentivePooling).
        """
        self.checkpoint = checkpoint
        self.cache_dir = cache_dir
        self.num_channels = num_channels
        super().__init__(
            num_classes=num_classes,
            embedding_size=embedding_size,
            classifier=classifier,
            freeze_backbone=freeze_backbone,
            local_checkpoint=local_checkpoint,
            load_classifier_checkpoint=load_classifier_checkpoint,
            preprocess_in_model=preprocess_in_model,
            pretrain_info=pretrain_info,
            pooling=pooling,
        )
        self.restrict_logits = restrict_logits  # This is just for logging purposes as restriction is handled in base_module.py
        self.config = self.model.config

        if self.pooling == "default":
            self.pooler = nn.LayerNorm(
                self.config.hidden_sizes[-1], eps=self.config.layer_norm_eps
            )
        elif self.pooling == "attentive":
            self.pooler = AttentivePooling(dim=embedding_size, num_heads=8)
        elif self.pooling == "average":
            self.pooler = AveragePooling()

        if local_checkpoint:
            self._load_local_checkpoint()

    def _load_model(self) -> ConvNextModel:
        adjusted_state_dict = None

        if self.checkpoint:
            if self.local_checkpoint:
                state_dict = torch.load(self.local_checkpoint)["state_dict"]

                # Update this part to handle the necessary key replacements
                adjusted_state_dict = {}
                for key, value in state_dict.items():
                    # Handle 'model.model.' prefix
                    new_key = key.replace("model.model.", "")

                    # Handle 'model._orig_mod.model.' prefix
                    new_key = new_key.replace("model._orig_mod.model.", "")

                    # Assign the adjusted key
                    adjusted_state_dict[new_key] = value

            return ConvNextForImageClassification.from_pretrained(
                self.checkpoint,
                num_labels=self.num_classes,
                num_channels=self.num_channels,
                cache_dir=self.cache_dir,
                state_dict=adjusted_state_dict,
                ignore_mismatched_sizes=True,
            )
        else:
            config = AutoConfig.from_pretrained(
                "facebook/convnext-base-224-22k",
                num_labels=self.num_classes,
                num_channels=self.num_channels,
            )
            return ConvNextForImageClassification(config)

    def _load_preprocessor(self) -> nn.Module:
        """
        Loads the preprocessor for the ConvNext model.
        This method is used to preprocess the input audio data before passing it to the model.
        """
        return nn.Sequential(
            torchaudio.transforms.Spectrogram(n_fft=1024, hop_length=320, power=2.0),
            torchaudio.transforms.MelScale(n_mels=128, n_stft=513, sample_rate=32_000),
            transforms.Normalize((-4.268,), (4.569,)),
            PowerToDB(top_db=80),
        )

    def get_embeddings(self, input_tensor) -> torch.Tensor:
        output = self.model(input_tensor, output_hidden_states=True, return_dict=True)
        if self.pooling == "default":
            embeddings = self.pooler(output.hidden_states[-1].mean([-2, -1]))
        elif self.pooling == "attentive" or self.pooling == "average":
            # transform (B, N, H, W) to (B, N, C)
            x = output.hidden_states[-1].flatten(2).transpose(1, 2)
            embeddings = self.pooler(x)
        return embeddings

    def forward(
        self, input_values: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Defines the forward pass of the ConvNext model.

        Args:
            input_values (torch.Tensor): An input batch.
            labels (Optional[torch.Tensor]): The corresponding labels. Default is None.

        Returns:
            torch.Tensor: The output of the ConvNext model.
        """
        if self.preprocess_in_model:
            input_values = self.preprocess(input_values)
        if self.classifier is not None:
            embeddings = self.get_embeddings(input_values)
            logits = self.classifier(embeddings)
        else:
            output = self.model(input_values)
            logits = output.logits

        return logits

    @torch.inference_mode()
    def get_logits(self, dataloader, device):
        pass

    @torch.inference_mode()
    def get_probas(self, dataloader, device):
        pass

    @torch.inference_mode()
    def get_representations(self, dataloader, device):
        pass


class ConvNextEmbedding(nn.Module):
    """
    ConvNext model for audio classification.
    """

    MEAN = -4.2677393
    STD = 4.5689974

    def __init__(
        self,
        num_channels: int = 1,
        num_classes: Optional[int] = None,
        checkpoint: Optional[str] = None,
        local_checkpoint: Optional[str] = None,
        cache_dir: Optional[str] = None,
        pretrain_info: PretrainInfoConfig = None,
    ):
        """
        Note: Either num_classes or pretrain_info must be given
        Args:
            num_channels: Number of input channels.
            checkpoint: huggingface checkpoint path of any model of correct type
            local_checkpoint: local path to checkpoint file
            cache_dir: specified cache dir to save model files at
            pretrain_info: hf_path and hf_name of info will be used to infer if num_classes is None
        """
        super().__init__()

        if pretrain_info:
            self.hf_path = pretrain_info.hf_path
            self.hf_name = (
                pretrain_info.hf_name
                if not pretrain_info.hf_pretrain_name
                else pretrain_info.hf_pretrain_name
            )
        else:
            self.hf_path = None
            self.hf_name = None

        self.num_channels = num_channels
        self.checkpoint = checkpoint
        self.local_checkpoint = local_checkpoint
        self.cache_dir = cache_dir

        self.model = None

        self._initialize_model()

    def _initialize_model(self):
        """Initializes the ConvNext model based on specified attributes."""

        adjusted_state_dict = None

        if self.checkpoint:
            if self.local_checkpoint:
                state_dict = torch.load(self.local_checkpoint)["state_dict"]

                # Update this part to handle the necessary key replacements
                adjusted_state_dict = {}
                for key, value in state_dict.items():
                    # Handle 'model.model.' prefix
                    new_key = key.replace("model.model.", "")

                    # Handle 'model._orig_mod.model.' prefix
                    new_key = new_key.replace("model._orig_mod.model.", "")

                    # Assign the adjusted key
                    adjusted_state_dict[new_key] = value

            self.model = ConvNextModel.from_pretrained(
                self.checkpoint,
                num_channels=self.num_channels,
                cache_dir=self.cache_dir,
                state_dict=adjusted_state_dict,
                ignore_mismatched_sizes=True,
            )

        else:
            print("Using pretrained convnext")
            config = AutoConfig.from_pretrained(
                "facebook/convnext-base-224-22k",
                num_channels=self.num_channels,
            )
            self.model = ConvNextModel(config)

    def preprocess(
        self, input_values: torch.Tensor, input_tdim=500, sample_rate=32000
    ) -> torch.Tensor:
        """
        Preprocesses the input values by applying mel-filterbank transformation.
        Args:
            input_values (torch.Tensor): Input tensor of shape (batch_size, num_samples).
            input_tdim (int): The number of frames to keep. Defaults to 500.
            sample_rate (int): The sampling rate of the input tensor. Defaults to 16000.
        Returns:
            torch.Tensor: Preprocessed tensor of shape (batch_size, 1, num_mel_bins, num_frames).
        """
        device = input_values.device
        melspecs = []
        for waveform in input_values:
            melspec = kaldi.fbank(
                waveform,
                htk_compat=True,
                window_type="hanning",
                num_mel_bins=128,
                use_energy=False,
                sample_frequency=sample_rate,
                frame_shift=10,
            )  # shape (n_frames, 128)
            # print(melspec.shape)
            if melspec.shape[0] < input_tdim:
                melspec = F.pad(melspec, (0, 0, 0, input_tdim - melspec.shape[0]))
            else:
                melspec = melspec[:input_tdim]
            melspecs.append(melspec)
        melspecs = torch.stack(melspecs).to(device)
        melspecs = melspecs.unsqueeze(1)  # shape (batch_size, 1, 128, 1024)
        melspecs = (melspecs - self.MEAN) / (self.STD * 2)
        return melspecs

    def get_embeddings(self, input_tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        input_tensor = self._preprocess(input_tensor)
        input_tensor = input_tensor.transpose(2, 3)
        output = self.model(input_tensor, output_hidden_states=True, return_dict=True)
        return output.pooler_output, None
