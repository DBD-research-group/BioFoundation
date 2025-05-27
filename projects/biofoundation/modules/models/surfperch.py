from typing import Optional, Literal
import torch
from torch import nn


from ml_collections import config_dict
from chirp.inference import embed_lib
from biofoundation.modules.models.birdset_model import BirdSetModel

from birdset.utils import pylogger

log = pylogger.get_pylogger(__name__)


class SurfPerchModel(BirdSetModel):
    """
    Add
    """

    EMBEDDING_SIZE = 768

    def __init__(
        self,
        num_classes: int | None,
        embedding_size: int = EMBEDDING_SIZE,
        local_checkpoint: str = None,
        checkpoint_path: str = '',
        load_classifier_checkpoint: bool = True,
        freeze_backbone: bool = False,
        preprocess_in_model: bool = True,
        classifier: nn.Module | None = None,
        pretrain_info = None,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            embedding_size=embedding_size,
            local_checkpoint=local_checkpoint,
            load_classifier_checkpoint=load_classifier_checkpoint,
            freeze_backbone=freeze_backbone,
            preprocess_in_model=preprocess_in_model,
            pretrain_info=pretrain_info,
        )
        self.model = None  # Placeholder for the loaded model
        self.checkpoint_path = checkpoint_path
        self.load_model()

        if classifier is None:
            self.classifier = nn.Linear(embedding_size, num_classes)
        else:
            self.classifier = classifier

        if local_checkpoint:
            self._load_local_checkpoint()

        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

    def load_model(self) -> None:
        """
        Load the model from shared storage.
        """
        # Model specific parameters: PLEASE DO NOT CHANGE THE CODE IN THIS CELL.
        config = config_dict.ConfigDict()
        embed_fn_config = config_dict.ConfigDict()
        embed_fn_config.model_key = 'taxonomy_model_tf'
        model_config = config_dict.ConfigDict()

        # The size of each "chunk" of audio.
        model_config.window_size_s = 5.0

        # The hop size
        model_config.hop_size_s = 5.0

        # All audio in this tutorial is resampled to 32 kHz.
        model_config.sample_rate = 32000

        # The location of the pre-trained model.
        model_config.model_path = self.checkpoint_path

        # Only write embeddings to reduce size. The Perch codebase supports serializing
        # a variety of metadata along with the embeddings, but for the purposes of this
        # tutorial we will not need to make use of those features.
        embed_fn_config.write_embeddings = True
        embed_fn_config.write_logits = False
        embed_fn_config.write_separated_audio = False
        embed_fn_config.write_raw_audio = False

        config.embed_fn_config = embed_fn_config
        embed_fn_config.model_config = model_config

        # These two settings can be used to break large inputs up into smaller chunks;
        # this is especially helpful for dealing with long files or very large datasets.
        # Given free colab has limited resources, you may want to reduce shard_len_s to 
        # 10 to prevent system RAM from becoming overloaded.
        config.shard_len_s = 60 # 
        config.num_shards_per_file = -1

        # Number of parent directories to include in the filename. This allows us to
        # process raw audio that lives in multiple directories.
        config.embed_fn_config.file_id_depth = 1

        # If your dataset is large its useful to split the TFRecords across multiple
        # shards so I/O operations can be parallized.
        config.tf_record_shards = 10


        embed_fn = embed_lib.EmbedFn(**config.embed_fn_config)
        print('\n\nLoading model(s)...')
        embed_fn.setup()

        print('\n\nTest-run of model...')
        z = np.zeros([int(model_config.sample_rate * model_config.window_size_s)])
        embed_fn.embedding_model.embed(z)
        print('Setup complete!')

        self.model = embed_fn

    def _preprocess(self, input_values: torch.Tensor) -> torch.Tensor:
        """
        Preprocessing for the input values is done in BETAs.py
        The waveform gets resampled to 16kHz, transformed into a fbank and then normalized.
        """
        return input_values

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
        embeddings = self.get_embeddings(input_values)
        # flattend_embeddings = embeddings.reshape(embeddings.size(0), -1)
        return self.classifier(embeddings)

    def get_embeddings(self, input_values: torch.Tensor) -> torch.Tensor:
        """
        Get the embeddings and logits from the BEATs model.

        Args:
            input_tensor (torch.Tensor): The input tensor for the model.

        Returns:
            torch.Tensor: The embeddings from the model.
        """
#@title { vertical-output: true }
# Embed! This step may take several minutes to run.
        # Convert torch tensor to numpy array if needed
        if isinstance(input_values, torch.Tensor):
            audio = input_values.detach().cpu().numpy()
        else:
            audio = input_values

        # Use the embed_fn from the original model
        # You may need to ensure embed_fn, model_config, and config are available as attributes
        file_id = "dummy_id"
        offset_s = 0.0

        # Generate the Example proto
        example = self.embed_fn.audio_to_example(file_id, offset_s, audio)
        if example is None:
            raise RuntimeError("Failed to generate embedding for input audio.")

        # Parse the example to get the embedding
        parsed = self.embed_fn.parse_example(example)
        embedding = parsed['embedding']

        # Optionally apply pooling if provided
        if pooling is not None:
            embedding = pooling(embedding)

        # Convert to torch tensor if needed
        embedding_tensor = torch.from_numpy(embedding).float()
        return embedding_tensor