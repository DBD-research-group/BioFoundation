from birdset.datamodule.components.transforms import BirdSetTransformsWrapper
from .base_datamodule import BaseDataModuleHF, DatasetConfig, LoadersConfig
from datasets import load_dataset, IterableDataset, IterableDatasetDict, DatasetDict, Audio, Dataset
import logging

class BEANSDataModule(BaseDataModuleHF):
    def __init__(
            self,
            dataset: DatasetConfig = DatasetConfig(),
            loaders: LoadersConfig = LoadersConfig(),
            transforms: BirdSetTransformsWrapper = BirdSetTransformsWrapper(),
            mapper: None = None
    ):
        super().__init__(
            dataset=dataset,
            loaders=loaders,
            transforms=transforms,
            mapper=mapper
        )

    def _preprocess_data(self, dataset):
        """
        Preprocess the data. For BEANS we will just return the dataset as it is.
        """


        return dataset



    def _load_data(self,decode: bool = True):    
        """
        Load audio dataset from Hugging Face Datasets. For BEANS the audio column is named path so we will rename it to audio and we have to do one-hot encoding for the labels.

        Returns HF dataset with audio column casted to Audio feature, containing audio data as numpy array and sampling rate.
        """
        logging.info("> Loading data set.")
        dataset = load_dataset(
            name=self.dataset_config.hf_name,
            path=self.dataset_config.hf_path,
            cache_dir=self.dataset_config.data_dir,
            num_proc=3,
        )
        # Leave it here just in case
        if isinstance(dataset, IterableDataset |IterableDatasetDict):
            logging.error("Iterable datasets not supported yet.")
            return
        assert isinstance(dataset, DatasetDict | Dataset)
        dataset = self._ensure_train_test_splits(dataset)

        if self.dataset_config.subset:
            dataset = self._fast_dev_subset(dataset, self.dataset_config.subset)

        # Rename some columns and remove unnamed column (Leftover from BEANS processing)
        dataset = dataset.rename_column("path", "audio")
        dataset = dataset.rename_column("label", "labels")
        dataset = dataset.remove_columns('Unnamed: 0')

        # Then we have to map the label to integers if they are strings (One-hot encoding)
        labels = set()
        for split in dataset.keys():
            labels.update(dataset[split]["labels"])

        label_to_id = {lbl: i for i, lbl in enumerate(labels)}

        def label_to_id_hot(row):
            l = [0.0] * len(labels)
            l[label_to_id[row["labels"]]] = 1.0
            row["labels"] = l
            return row

        def label_to_id_fn(row):
            row["labels"] = label_to_id[row["labels"]]
            return row

        dataset = dataset.map(label_to_id_fn)
        
        # Normal casting
        dataset = dataset.cast_column(
            column="audio",
            feature=Audio(
                sampling_rate=self.dataset_config.sampling_rate,
                mono=True,
                decode=decode,
            ),
        )
        
        return dataset