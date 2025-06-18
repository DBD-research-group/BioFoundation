from typing import Literal, Optional
from biofoundation.modules.models.Pooling import (
    AttentivePooling,
    AttentivePooling_old,
    AveragePooling,
)
from biofoundation.modules.models.vit import ViT
import torch
from torch import nn
import torch.nn.functional as F
from torchaudio.compliance.kaldi import fbank

from birdset.configs.model_configs import PretrainInfoConfig
from timm.models.vision_transformer import VisionTransformer, PatchEmbed


class BirdMAEModel(ViT):
    """
    BirdMAE model from the paper "Can Masked Autoencoders Also Listen to Birds?"
    Rauch et al. 2025 https://arxiv.org/abs/2504.12880
    """

    EMBEDDING_SIZE = 1024
    MEAN = -4.2677393
    STD = 4.5689974

    def __init__(
        self,
        num_classes: int,
        embedding_size: int = EMBEDDING_SIZE,
        checkpoint_path: str = "/workspace/models/birdmae/2025-01-13_213828_AudioMAE_XCL_epoch=149.ckpt",
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
        self.pooling_type = pooling
        if self.pooling_type == "attentive":
            attentive_heads = (
                embedding_size // 64
            )  # embedding_size // 64 should be 12 for 768
            self.attentive_pooling = AttentivePooling(
                dim=embedding_size, num_heads=attentive_heads
            )
        elif self.pooling_type == "attentive_old":
            attentive_heads = embedding_size // 8  # beats uses 8 heads
            self.attentive_pooling = AttentivePooling_old(
                embed_dim=embedding_size, num_heads=attentive_heads
            )
        elif self.pooling_type == "average":
            self.average_pooling = AveragePooling()

    def _load_model(self) -> None:
        vit = VisionTransformer(
            img_size=(512, 128),
            patch_size=16,
            in_chans=1,
            num_classes=9735,
            embed_dim=1024,
            depth=24,
            num_heads=16,
            mlp_ratio=4.0,
            qkv_bias=True,
            norm_layer=torch.nn.LayerNorm,
        )

        pretrained_weights_path = self.checkpoint_path
        img_size = (512, 128)
        embed_dim = 1024
        num_patches = 256  # birdset
        vit.patch_embed = PatchEmbed(img_size, 16, 1, embed_dim)
        # self.patch_embed = PatchEmbed_org(img_size, 16, 1, self.embed_dim)
        vit.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False
        )  # to load pretrained pos embed
        try:
            pre_state_dict = torch.load(pretrained_weights_path, map_location="cpu")[
                "model"
            ]
        except:
            pre_state_dict = torch.load(
                pretrained_weights_path, map_location="cpu", weights_only=False
            )["state_dict"]
        pretrained_state_dict = {}
        for key, value in pre_state_dict.items():
            if key.startswith("decoder."):
                # Skip any key that starts with "decoder."
                continue
            elif key.startswith("encoder."):
                # Remove the "encoder." prefix
                new_key = key[len("encoder.") :]
            else:
                # Use the original key if no prefix
                new_key = key

            # Add the modified key-value pair to the new state dict
            pretrained_state_dict[new_key] = value

        info = vit.load_state_dict(pretrained_state_dict, strict=False)
        # patch_hw = (img_size[1] // 16, img_size[0] // 16) # 16=patchsize
        # #patch_hw = (img_size[0] // 16, img_size[1] // 16)
        # pos_embed = get_2d_sincos_pos_embed_flexible(self.pos_embed.size(-1), patch_hw, cls_token=True) # not trained, overwrite from sincos
        # self.pos_embed.data = torch.from_numpy(pos_embed).float().unsqueeze(0)
        # print("Loaded pretrained weights with info:", info)
        return vit

    def _pad_and_normalize(self, fbank_features):
        difference = 512 - fbank_features[0].shape[0]
        min_value = fbank_features.min()
        # min_value = -80
        if 512 > fbank_features.shape[0]:
            padding = (0, 0, 0, difference)
            fbank_features = F.pad(fbank_features, padding, value=min_value.item())
        return fbank_features

    def _preprocess(self, input_values: torch.Tensor) -> torch.Tensor:
        """
        Preprocess the input values for the model.

        Args:
            input_values (torch.Tensor): The input tensor to preprocess.

        Returns:
            torch.Tensor: The preprocessed input tensor.
        """
        fbank_features = [
            fbank(
                waveform,
                htk_compat=True,
                sample_frequency=32_000,
                use_energy=False,
                window_type="hanning",
                num_mel_bins=128,
                dither=0.0,
                frame_shift=10,
            )
            for waveform in input_values
        ]
        preprpocessed_fbank_features = torch.stack(fbank_features)
        # Pad and normalize the fbank features
        preprpocessed_fbank_features = self._pad_and_normalize(
            preprpocessed_fbank_features
        )
        # normalize the fbank features mean: -7.2 std: 4.43
        preprpocessed_fbank_features = (
            preprpocessed_fbank_features - self.MEAN
        ) / self.STD
        return preprpocessed_fbank_features.unsqueeze(1)  # Add channel dimension

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
        )  # shape (batch_size, 257, 1024)
        return self.pool(embeddings, self.pooling_type)

    def get_num_layers(self) -> int:
        """
        Get the number of layers in the model.

        Returns:
            int: The number of layers in the model.
        """
        return len(self.model.encoder.layers)
