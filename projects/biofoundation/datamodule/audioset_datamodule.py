from birdset.datamodule.components.transforms import BirdSetTransformsWrapper
from birdset.datamodule.base_datamodule import BaseDataModuleHF
from birdset.configs import DatasetConfig, LoadersConfig
from datasets import (
    load_dataset,
    IterableDataset,
    IterableDatasetDict,
    DatasetDict,
    Audio,
    Dataset,
)
from birdset.utils import pylogger
import logging

log = pylogger.get_pylogger(__name__)



class AS20DataModule(BaseDataModuleHF):
    def __init__(
        self,
        dataset: DatasetConfig = DatasetConfig(),
        loaders: LoadersConfig = LoadersConfig(),
        transforms: BirdSetTransformsWrapper = BirdSetTransformsWrapper(),
    ):
        super().__init__(dataset=dataset, loaders=loaders, transforms=transforms)


    def _load_data(self, decode: bool = True) -> DatasetDict:
        """
        Load audio dataset from Hugging Face Datasets. Same as the base datamodule method but num_proc set to 1.

        Returns HF dataset with audio column casted to Audio feature, containing audio data as numpy array and sampling rate.
        """
        log.info("> Loading data set.")

        dataset_args = {
            "path": self.dataset_config.hf_path,
            "cache_dir": self.dataset_config.data_dir,
            "num_proc": 1,
            "trust_remote_code": True,
        }

        dataset = load_dataset(**dataset_args)
        if isinstance(dataset, IterableDataset | IterableDatasetDict):
            log.error("Iterable datasets not supported yet.")
            return
        assert isinstance(dataset, DatasetDict | Dataset)
        dataset = self._ensure_train_test_splits(dataset)

        if self.dataset_config.subset:
            dataset = self._fast_dev_subset(dataset, self.dataset_config.subset)

        dataset = dataset.rename_column("wav", "audio")

        dataset = dataset.cast_column(
            column="audio",
            feature=Audio(
                sampling_rate=self.dataset_config.sampling_rate,
                mono=True,
                decode=decode,
            ),
        )
        return dataset    

    def _preprocess_data(self, dataset):
        """
        Preprocess the data
        """

        dataset = dataset.rename_column("json", "labels")
        dataset = dataset.map(
            lambda x: {"labels": x["labels"]["label"]},
            remove_columns=["labels"],
            load_from_cache_file=True,
            num_proc=self.dataset_config.n_workers,
        )

        dataset = dataset.map(
                self._classes_one_hot,
                batched=True,
                batch_size=200,
                load_from_cache_file=True,
                num_proc=self.dataset_config.n_workers,
            )
        return dataset
