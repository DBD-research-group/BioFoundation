# from biofoundation.modules.models.birdset_model import BirdSetModel
from torch import nn
import torchvision.models as models
import torch
import torch.nn.functional as F
from torchaudio.compliance import kaldi
import torchaudio.transforms as T
import torchvision.transforms as TV
from biofoundation.modules.models.biofoundation_model import BioFoundationModel


class iNatSoundModel(BioFoundationModel):
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
        spectograms = self.preprocess(input_values)
        return self.model(spectograms)

    #! Seems very similar to the other implementation except the chunking and frame_length and frame_shift which are bit different in the def values
    def preprocess_old(self, input_values: torch.Tensor) -> torch.Tensor:
        TARGET_SR = 22050
        FRAME_LENGTH = 512 * 1000 / TARGET_SR  # in ms
        FRAME_SHIFT = 128 * 1000 / TARGET_SR

        device = input_values.device
        melspecs = []
        target_length = 512

        for waveform in input_values:
            if (
                waveform.shape[-1] < 512
            ):  #! Does this make sense? But shouldnt be a problem anyway
                waveform = F.pad(waveform, (0, 512 - waveform.shape[-1]))
            melspec = kaldi.fbank(
                waveform,
                frame_length=FRAME_LENGTH,
                frame_shift=FRAME_SHIFT,
                window_type="hanning",
                low_freq=50,
                high_freq=11025,
                sample_frequency=22050,
                num_mel_bins=128,
            )
            print(melspec.shape)

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

    def preprocess(self, batch_waveforms, sample_rate=22050):
        TARGET_SR = 22050
        WINDOW_DURATION = 3.0
        WINDOW_SIZE = int(TARGET_SR * WINDOW_DURATION)
        STRIDE_SIZE = WINDOW_SIZE // 2
        MEL_BINS = 128
        MEL_LOW = 50
        MEL_HIGH = TARGET_SR // 2
        FRAME_LENGTH = 512 * 1000 / TARGET_SR  # in ms
        FRAME_SHIFT = 128 * 1000 / TARGET_SR
        TARGET_SIZE = (224, 224)

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
            db -= db.min()
            db /= db.max()
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
