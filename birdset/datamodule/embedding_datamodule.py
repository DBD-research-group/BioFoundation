from birdset.datamodule.components.transforms import EmbeddingTransforms
from birdset.datamodule.base_datamodule import BaseDataModuleHF
from birdset.configs import NetworkConfig, DatasetConfig, LoadersConfig
from datasets import DatasetDict, Dataset, concatenate_datasets
from collections import defaultdict
from tabulate import tabulate
from birdset.utils import pylogger
from tqdm import tqdm
from dataclasses import dataclass
from typing import Union
import torch
import torchaudio
import os


log = pylogger.get_pylogger(__name__)

@dataclass
class EmbeddingModuleConfig(NetworkConfig):
    """
    A dataclass that makes sure the model inherits from EmbeddingClassifier.

    """
    model: Union[torch.nn.Module] = None # Model for extracting the embeddings

class EmbeddingDataModule(BaseDataModuleHF):
    def __init__(
            self,
            dataset: DatasetConfig = DatasetConfig(),
            loaders: LoadersConfig = LoadersConfig(),
            transforms: EmbeddingTransforms = EmbeddingTransforms(),
            k_samples: int = 0,
            val_batches: int = None, # Should val set be created
            test_ratio: float = 0.5, # Ratio of test set if val set is also created
            low_train: bool = False, # If low train set is used
            embedding_model: EmbeddingModuleConfig = EmbeddingModuleConfig(),
            average: bool = True,
            gpu_to_use: int = 0
    ):
        """
        DataModule for extracting embeddings as a preprocessing step from audio data.

        Args:
            dataset (DatasetConfig, optional): The config of the dataset to use. Defaults to DatasetConfig().
            loaders (LoadersConfig, optional): Loader config. Defaults to LoadersConfig().
            transforms (BirdSetTransformsWrapper, optional): uses EmbeddingsTransform so no Embeddings are cut off. Defaults to EmbeddingTransforms().
            k_samples (int, optional): The amount of samples per class that should be used. Defaults to 0 where the predefined sets are used.
            val_batches (int, optional): If a validation set should be used or not. Defaults to None which means that the normal validation split or amount is used.
            test_ratio (float, optional): Ratio of test set if val set is also created. Defaults to 0.5.
            low_train (bool, optional): If low train set is used (Exists in BEANS for example). Defaults to False.
            embedding_model (EmbeddingModuleConfig, optional): Model for extracting the embeddings. Defaults to EmbeddingModuleConfig().
            average (bool, optional): If embeddings should be averaged if the audio clip is too long. Defaults to True.
            gpu_to_use (int, optional): Which GPU should be used for extracting the embeddings. Defaults to 0.
        """
        super().__init__(
            dataset=dataset,
            loaders=loaders,
            transforms=transforms
        )
        self.device = torch.device(f'cuda:{gpu_to_use}')
        self.k_samples = k_samples
        self.val_batches = val_batches
        self.test_ratio = test_ratio
        self.low_train = low_train
        self.average = average
        self.id_to_label = defaultdict(str)
        self.embedding_model_name = embedding_model.model_name
        self.embedding_model = embedding_model.model.to(self.device) # Move Model to GPU
        self.embedding_model.eval()  # Set the model to evaluation mode
        self.sampling_rate = embedding_model.sampling_rate
        self.max_length = embedding_model.length
        self.disk_save_path = os.path.join(
            self.dataset_config.data_dir,
            f"{self.dataset_config.dataset_name}_processed_{self.dataset_config.seed}_{self.embedding_model_name}_{self.k_samples}_{self.average}_{self.val_batches}_{self.low_train}_{self.sampling_rate}_{self.max_length}",
        )
        log.info(f"Using embedding model:{embedding_model.model_name} (Sampling Rate:{self.sampling_rate}, Window Size:{self.max_length})")

    def prepare_data(self):
        """
        Same as prepare_data in BaseDataModuleHF but checks if path exists and skips rest otherwise
        """
        if not self._prepare_done and os.path.exists(self.disk_save_path):
            #! We need to set self.len_trainset otherwise base_module doesn't work so we load train split to get the length
            self.len_trainset = len(self._get_dataset("train"))
            self._prepare_done = True

        log.info("Check if preparing has already been done.")
        if self._prepare_done:
            log.info("Skip preparing.")
            return

        log.info("Prepare Data")

        dataset = self._load_data()
        dataset = self._preprocess_data(dataset)
        dataset = self._create_splits(dataset)

        # set the length of the training set to be accessed by the model
        self.len_trainset = len(dataset["train"])
        self._save_dataset_to_disk(dataset)

        # set to done so that lightning does not call it again
        self._prepare_done = True

    def _preprocess_data(self, dataset):
        """
        Preprocess the data. This calls the _ksamples function to select k samples per class and calls the embedding extraction function. If multilabel is the task we will one hot encode. 
        """

        # Check if actually a dict
        dataset = self._ksamples(dataset)
        dataset = self._compute_embeddings(dataset)

        if self.dataset_config.task == 'multilabel':
            log.info(">> One-hot-encode classes")
            dataset = dataset.map(
                self._classes_one_hot,
                batched=True,
                batch_size=500,
                load_from_cache_file=True,
                num_proc=self.dataset_config.n_workers,
            )

        return dataset


    def _ksamples(self, dataset):
        """
        Use k_samples > 0 if you want control over amount of samples per class. The rest is used for validation and testing.
        If test_ratio == 1 then no validation set even if k_samples == 0!
        """
        if self.k_samples > 0:
            log.info(f">> Selecting {self.k_samples} Samples per Class this may take a bit...")
            merged_data = concatenate_datasets([dataset['train'], dataset['valid'], dataset['test']])

            # Shuffle the merged data
            merged_data.shuffle() #TODO: Check if this is affected by the public seed
            
            # Create a dictionary to store the selected samples per class
            selected_samples = defaultdict(list)
            train_count = defaultdict(int)
            testval_count = defaultdict(int)
            rest_samples = []
            # Iterate over the merged data and select the desired number of samples per class
            for sample in tqdm(merged_data, total=len(merged_data), desc="Selecting samples"):
                label = sample['labels']
                if len(selected_samples[label]) < self.k_samples:
                    selected_samples[label].append(sample)
                    train_count[label] += 1
                else:
                    rest_samples.append(sample)
                    testval_count[label] += 1    

            
            # Create and print table to show class distribution
            headers = ["Class", "#Train-Samples", "#Test,Valid-Samples"]
            rows = []
            
            for class_id in selected_samples.keys():
                rows.append([self.id_to_label[class_id], train_count[class_id], testval_count[class_id]])
            
            print(tabulate(rows, headers, tablefmt="rounded_grid"))
            
            # Flatten the selected samples into a single list
            selected_samples = [sample for samples in selected_samples.values() for sample in samples]

            # Split the selected samples into training, validation, and testing sets

            if self.val_batches == 0:
                train_data = selected_samples
                test_data = rest_samples
                val_data = Dataset.from_dict({})
            
            else:    
                num_samples = len(rest_samples)
                num_test_samples = int(self.test_ratio * num_samples)

                train_data = selected_samples
                test_data = rest_samples[:num_test_samples]
                val_data = rest_samples[num_test_samples:]
                val_data = Dataset.from_dict({key: [sample[key] for sample in val_data] for key in val_data[0]}) #! Use first test sample as val cant be empty
            
            train_data = Dataset.from_dict({key: [sample[key] for sample in train_data] for key in train_data[0]})
            test_data = Dataset.from_dict({key: [sample[key] for sample in test_data] for key in test_data[0]})

            # Combine into a DatasetDict
            dataset = DatasetDict({
                'train': train_data,
                'valid': val_data,
                'test': test_data
            })
        else:
            if self.val_batches == 0:
                dataset['test'] = concatenate_datasets([dataset['valid'], dataset['test']])
                dataset['valid'] = Dataset.from_dict({}) #! So that no split is created in the dataloader
                
            if self.low_train:
                del dataset['train']
                dataset['train'] = dataset['train_low']

            del dataset['train_low']    
                
            
            
        return dataset




    def _compute_embeddings(self, dataset):
        """
        Compute Embeddings for the entire dataset and update the dataset in place.
        """
        # Define the function that will be applied to each sample
        def compute_and_update_embedding(sample):
            with torch.no_grad():
                # Get the embedding for the audio sample
                embedding = self._get_embedding(sample['audio'])
                # Update the sample with the new embedding
                sample['embedding'] = {}
                sample['embedding']['array'] = embedding.squeeze(0).cpu().numpy()
                sample.pop('audio') # Remove audio to save space
            return sample

        # Apply the transformation to each split in the dataset
        for split in dataset.keys():
            log.info(f">> Extracting Embeddings for {split} Split")
            # Apply the embedding function to each sample in the split
            dataset[split] = dataset[split].map(compute_and_update_embedding, desc="Extracting Embeddings")

        return dataset
        

    def _get_embedding(self, audio):
        # Get waveform and sampling rate
        waveform = torch.tensor(audio['array'], dtype=torch.float32).to(self.device) # Get waveform audio and move to GPU
        dataset_sampling_rate = audio['sampling_rate']
        # Resample audio
        audio = self._resample_audio(waveform, dataset_sampling_rate)
        
        # Zero-padding
        audio = self._zero_pad(waveform)

        # Check if audio is too long 
        if waveform.shape[0] > self.max_length * self.sampling_rate:
            if self.average:
                return self._frame_and_average(waveform) 
            else:
                audio = audio[:self.max_length * self.sampling_rate]
                return self.embedding_model.get_embeddings(audio.view(1, 1, -1))[0] 
        else:
            return self.embedding_model.get_embeddings(audio.view(1, 1, -1))[0]

    # Resample function
    def _resample_audio(self, audio, orig_sr):
        resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=self.sampling_rate)
        return resampler(audio)

    # Zero-padding function
    def _zero_pad(self, audio):
        desired_num_samples = self.max_length * self.sampling_rate 
        current_num_samples = audio.shape[0]
        padding = desired_num_samples - current_num_samples
        if padding > 0:
            #print('padding')
            pad_left = padding // 2
            pad_right = padding - pad_left
            audio = torch.nn.functional.pad(audio, (pad_left, pad_right))
        return audio

    # Average multiple embeddings function
    def _frame_and_average(self, audio):
        # Frame the audio
        frame_size = self.max_length * self.sampling_rate
        hop_size = self.max_length * self.sampling_rate
        frames = audio.unfold(0, frame_size, hop_size)
        
        # Generate embeddings for each frame
        l = []
        for frame in frames:
            embedding = self.embedding_model.get_embeddings(frame.view(1, 1, -1))[0] 
            l.append(embedding[0]) # To just use embeddings not logits
        
        embeddings = torch.stack(tuple(l))
        
        # Average the embeddings
        averaged_embedding = embeddings.mean(dim=0)
        
        return averaged_embedding        
    
    def _save_dataset_to_disk(self, dataset: Dataset | DatasetDict):
        """
        Saves the dataset to disk.

        This method sets the format of the dataset to numpy, prepares the path where the dataset will be saved, and saves
        the dataset to disk. If the dataset already exists on disk, it does not save the dataset again.

        Args:
            dataset (datasets.DatasetDict): The dataset to be saved. The dataset should be a Hugging Face `datasets.DatasetDict` object.

        Returns:
            None
        """
        
        #! Due to randomness somewhere in model,resampling,... the fingerprint isn't the same
        # For now now we add k_samples, model_name  (Potentially k_Samples could also be picked from extracted embeddings already but this is easier and it doesn't take that long)
        #! Manual deleting may be needed if you want to recompute the embeddings
        
        dataset.set_format("np") # Removes slight changes as dtype should be same
        if os.path.exists(self.disk_save_path):
            log.info(f"Train fingerprint found in {self.disk_save_path}, saving to disk is skipped")
        else:
            log.info(f"Saving to disk: {self.disk_save_path}")
            dataset.save_to_disk(self.disk_save_path)