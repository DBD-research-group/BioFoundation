# from biofoundation.modules.models.birdset_model import BirdSetModel
from torch import nn
import torchvision.models as models
import torch
import torch.nn.functional as F
from torchaudio.compliance import kaldi
import torchaudio.transforms as T
import torchvision.transforms as TV

# from biofoundation.modules.models.biofoundation_model import BioFoundationModel
from biofoundation.modules.models.vit import ViT
from birdset.configs.model_configs import PretrainInfoConfig
from typing import Literal, Optional, Tuple


class Vit_iNatSoundModel(ViT):
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
        pretrain_info: PretrainInfoConfig = None,
        pooling: Literal["just_cls", "attentive", "average"] = "just_cls",
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            embedding_size=embedding_size,
            local_checkpoint=local_checkpoint,
            load_classifier_checkpoint=load_classifier_checkpoint,
            preprocess_in_model=preprocess_in_model,
            pooling=pooling,
        )

        self.mel_spectrogram = T.MelSpectrogram(
            window_fn=torch.hann_window,
            sample_rate=22050,
            n_fft=1024,
            win_length=256,
            hop_length=32,
            f_min=50,
            f_max=11025,
            n_mels=298,
            power=1.0,
        )

        self.num_classes = num_classes
        # self.model = None
        # self.load_model()

        if classifier is None:
            self.model.heads.head = nn.Linear(embedding_size, num_classes)
        else:
            self.model.heads.head = classifier

        if local_checkpoint:
            self._load_local_checkpoint()

        if freeze_backbone:
            for name, param in self.model.named_parameters():
                if "heads.head" not in name:
                    param.requires_grad = False

    def _load_model(self) -> nn.Module:
        model = models.vit_b_16(weights=None)

        checkpoint_path = "/workspace/models/vit/vit_single_mixup.pt"
        state_dict = torch.load(checkpoint_path)

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
        self.mel_spectrogram = T.MelSpectrogram(
            window_fn=torch.hann_window,
            sample_rate=22050,
            n_fft=1024,
            win_length=256,
            hop_length=32,
            f_min=50,
            f_max=11025,
            n_mels=298,
            power=1.0,
        )

    def preprocess2(self, input_values: torch.Tensor) -> torch.Tensor:
        # faulty
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

    def preprocess(self, batch_waveforms, sample_rate=22050):
        TARGET_SR = 22050
        WINDOW_DURATION = 3.0
        WINDOW_SIZE = int(TARGET_SR * WINDOW_DURATION)
        STRIDE_SIZE = WINDOW_SIZE // 2
        MEL_BINS = 298
        MEL_LOW = 50
        MEL_HIGH = TARGET_SR // 2
        FRAME_LENGTH = 256  # * 1000 / TARGET_SR
        FRAME_SHIFT = 32  # 1000 / TARGET_SR
        TARGET_SIZE = (224, 224)
        MAX_DB_VALUE = 0.0
        MIN_DB_VALUE = -100.0
        N_FFT = 1024

        def resample_if_needed(waveform, orig_sr):  #! Brauch man nicht
            if orig_sr != TARGET_SR:
                resampler = T.Resample(orig_sr, TARGET_SR)
                waveform = resampler(waveform)
            return waveform

        def pad_or_trim(waveform, length):
            if waveform.size(-1) < length:
                pad = length - waveform.size(-1)
                waveform = torch.nn.functional.pad(waveform, (0, pad))
            return waveform[..., :length]

        def chunk_audio(waveform):
            total_len = waveform.size(-1)
            chunks = []
            if waveform.size(-1) <= WINDOW_SIZE:
                chunks = [pad_or_trim(waveform, WINDOW_SIZE)]
            else:
                for start in range(0, total_len, STRIDE_SIZE):
                    end = start + WINDOW_SIZE
                    chunk = pad_or_trim(waveform[..., start:end], WINDOW_SIZE)
                    chunks.append(chunk)
            return chunks

        def extract_features(wav_chunk):
            feats = T.MelSpectrogram(
                window_fn=torch.hann_window,
                sample_rate=TARGET_SR,
                n_fft=N_FFT,
                win_length=FRAME_LENGTH,
                hop_length=FRAME_SHIFT,
                f_min=MEL_LOW,
                f_max=MEL_HIGH,
                n_mels=MEL_BINS,
                power=1.0,
            )
            feats = kaldi.fbank(
                wav_chunk,
                num_mel_bins=MEL_BINS,
                sample_frequency=TARGET_SR,
                frame_length=FRAME_LENGTH,
                frame_shift=FRAME_SHIFT,
                dither=0.0,
                low_freq=MEL_LOW,
                high_freq=MEL_HIGH,
                use_energy=False,
                window_type="hanning",
            )
            return feats

        def log_mel_to_uint8(log_mel):
            db = 20 * torch.log10(torch.clamp(log_mel, min=1e-5))
            db -= MIN_DB_VALUE
            db /= MAX_DB_VALUE - MIN_DB_VALUE  # Normalize to [0, 1]
            img = (db * 255).clamp(0, 255).byte().T  # [mel, time]
            return img

        def format_for_vit(img_gray):
            img_rgb = img_gray.unsqueeze(0).repeat(3, 1, 1).float() / 255.0
            return TV.Resize(TARGET_SIZE, interpolation=TV.InterpolationMode.BILINEAR)(
                img_rgb
            )

        batch_size = batch_waveforms.size(0)
        all_chunks = []
        max_chunks = 0

        for waveform in batch_waveforms:
            # [C, T] → mono
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            if waveform.size(0) > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            waveform = resample_if_needed(waveform, sample_rate)
            chunks = chunk_audio(waveform)
            images = []
            for chunk in chunks:
                mel = extract_features(chunk)
                img_gray = log_mel_to_uint8(mel)
                img_rgb = format_for_vit(img_gray)
                images.append(img_rgb)
            max_chunks = max(max_chunks, len(images))
            all_chunks.append(images)

        # Pad chunks so all have same length
        for i in range(batch_size):
            num_chunks = len(all_chunks[i])
            if num_chunks < max_chunks:
                pad_img = torch.zeros((3, *TARGET_SIZE))
                all_chunks[i] += [pad_img] * (max_chunks - num_chunks)

        # batch_tensor = torch.stack([seq[0] for seq in all_chunks])  # [B, 3, 224, 224] We just take first chunk for now
        batch_tensor = torch.stack(
            [torch.stack(seq).mean(dim=0) for seq in all_chunks]
        )  # Average over chunks
        return batch_tensor

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
