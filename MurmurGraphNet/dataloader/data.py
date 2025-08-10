"""Dataset handlings.

Balanced sampler is supported for multi-label tasks.
"""

import os
import torch
import random
import librosa
import logging
import numpy as np
import pandas as pd
import torch.nn.functional as F

from pathlib import Path
from torch.utils.data import WeightedRandomSampler
from sklearn.preprocessing import MultiLabelBinarizer
from MurmurGraphNet.dataloader.sampler import BalancedRandomSampler, InfiniteSampler

class BaseRawAudioDataset(torch.utils.data.Dataset):
    def __init__(self, unit_samples, tfms=None, random_crop=False, return_filename=False):
        self.unit_samples = unit_samples
        self.tfms = tfms
        self.random_crop = random_crop
        self.return_filename = return_filename

    def __len__(self):
        raise NotImplementedError

    def get_audio(self, index):
        raise NotImplementedError

    def get_label(self, index):
        return None # implement me

    def __getitem__(self, index):
        label = self.get_label(index)

        if self.return_filename:
            fn = self.cfg.task_data + '/' + self.df.file_name.values[index]  # requires self.cfg & self.df to be set in advance.
            return fn if label is None else (fn, label)
        wav = self.get_audio(index) # shape is expected to be (self.unit_samples,)

        # Trim or stuff padding
        l = len(wav)
        if l > self.unit_samples:
            start = np.random.randint(l - self.unit_samples) if self.random_crop else 0
            wav = wav[start:start + self.unit_samples]
        elif l < self.unit_samples:
            wav = F.pad(wav, (0, self.unit_samples - l), mode='constant', value=0)
        wav = wav.to(torch.float)

        # Apply transforms
        if self.tfms is not None:
            wav = self.tfms(wav)

        # Return item
        return wav if label is None else (wav, label)

class WavDataset_paired(BaseRawAudioDataset):
    """
    Multi-position heart sound dataset returning (wav, label) format.
    wav shape: (5, unit_samples) where:
    - wav[0] = MV audio
    - wav[1] = AV audio  
    - wav[2] = PV audio
    - wav[3] = TV audio
    - wav[4] = mask channel (segmented mask info)
    """
    
    POSITIONS = ['MV', 'AV', 'PV', 'TV']
    
    def __init__(self, cfg, split, holdout_fold=None, always_one_hot=False, random_crop=True, 
                 classes=None, repeat_factor=10, missing_strategy='zero_fill'):
        super().__init__(cfg.unit_samples, tfms=None, random_crop=random_crop, 
                        return_filename=cfg.return_filename)
        self.cfg = cfg
        self.split = split
        self.repeat_factor = repeat_factor
        self.missing_strategy = missing_strategy
        
        # Load metadata
        df = pd.read_csv(cfg.task_metadata)
        df = df[df.split == split].reset_index(drop=True)
        print(f"Loaded {len(df)} records for {split} split")
        self.df = df
        
        # Build patient-position-segments mapping from CSV
        self._build_patient_segments_mapping()
        
        # Handle labels (get unique labels per patient)
        self._setup_labels(always_one_hot, classes)
        
        # Generate training samples
        self._generate_training_samples()
        
    def _build_patient_segments_mapping(self):
        """Build mapping from patient_id to {position: [file_paths]} from CSV"""
        
        self.patient_segments = {}
        self.patient_labels = {}
        
        for _, row in self.df.iterrows():
            file_name = row['file_name']  # e.g., 'train\\2530_AV_0.wav'
            label = row['label']
            
            # Parse filename: split\patient_id_position_segment.wav
            # Handle both Windows (\) and Unix (/) path separators
            file_name_clean = file_name.replace('\\', '/').split('/')[-1]  # Get just filename
            parts = file_name_clean.replace('.wav', '').split('_')
            
            if len(parts) < 3:
                logging.warning(f"Skipping file with unexpected format: {file_name}")
                continue
                
            patient_id, position = parts[0], parts[1]
            
            if position not in self.POSITIONS:
                logging.warning(f"Unknown position {position} in file: {file_name}")
                continue
            
            # Build full file path
            full_path = os.path.join(self.cfg.task_data, file_name.replace('\\', '/'))
            
            # Initialize dict structure if needed
            if patient_id not in self.patient_segments:
                self.patient_segments[patient_id] = {pos: [] for pos in self.POSITIONS}
            
            # Store in mapping
            self.patient_segments[patient_id][position].append(full_path)
            
            # Store patient label (will be the same for all segments of same patient)
            self.patient_labels[patient_id] = label
        
        # Log statistics
        print(f"Found {len(self.patient_segments)} patients in {self.split} split")
        position_stats = {pos: 0 for pos in self.POSITIONS}
        missing_stats = {pos: 0 for pos in self.POSITIONS}
        
        for patient_id, segments in self.patient_segments.items():
            for pos in self.POSITIONS:
                if segments[pos]:
                    position_stats[pos] += len(segments[pos])
                else:
                    missing_stats[pos] += 1
                    
        print("Position segments count:", position_stats)
        print("Missing position count:", missing_stats)
    
    def _setup_labels(self, always_one_hot, classes):
        """Setup label encoding"""
        # Get unique labels per patient
        patient_label_list = list(self.patient_labels.values())
        
        # Check if multi-label
        self.multi_label = any(',' in str(label) for label in patient_label_list)
        print("self.multi_label: ", self.multi_label)
        
        if self.multi_label or always_one_hot:
            # Multi-label encoding
            oh_enc = MultiLabelBinarizer()
            labels_split = [str(label).split(',') for label in patient_label_list]
            oh_enc.fit(labels_split)
            self.classes = oh_enc.classes_
            # Create patient_id -> encoded_label mapping
            self.patient_encoded_labels = {}
            for patient_id, label in self.patient_labels.items():
                encoded = oh_enc.transform([str(label).split(',')])[0]
                self.patient_encoded_labels[patient_id] = torch.tensor(encoded, dtype=torch.float32)
        else:
            # Single-label encoding
            unique_labels = sorted(set(patient_label_list))
            self.classes = unique_labels if classes is None else classes
            label_to_idx = {l: i for i, l in enumerate(self.classes)}
            
            # Create patient_id -> label_idx mapping
            self.patient_encoded_labels = {}
            for patient_id, label in self.patient_labels.items():
                self.patient_encoded_labels[patient_id] = torch.tensor(label_to_idx[label], dtype=torch.long)
                
    def _generate_training_samples(self):
        """Generate training samples: each sample = (patient_id, combination_id)"""
        self.samples = []
        
        for patient_id in self.patient_segments.keys():
            # Skip patients with no segments at all
            total_segments = sum(len(segs) for segs in self.patient_segments[patient_id].values())
            if total_segments == 0:
                continue
                
            # Skip patients with too many missing positions if using skip strategy
            if self.missing_strategy == 'skip_patient':
                available_positions = sum(1 for pos in self.POSITIONS 
                                       if self.patient_segments[patient_id][pos])
                if available_positions < 2:  # Require at least 2 positions
                    logging.warning(f"Skipping patient {patient_id}: only {available_positions} positions available")
                    continue
            
            # Generate multiple combinations for this patient
            for combo_id in range(self.repeat_factor):
                self.samples.append((patient_id, combo_id))
                
        print(f"Generated {len(self.samples)} training samples ({len(self.samples)//self.repeat_factor} unique patients)")

        self._build_labels_tensor()

    def _build_labels_tensor(self):
        """Build labels tensor for compatibility with existing code"""
        labels_list = []

        for patient_id, combo_id in self.samples:
            label_tensor = self.patient_encoded_labels[patient_id]
            labels_list.append(label_tensor)

        # Stack all labels into a single tensor
        if len(labels_list) > 0:
            self.labels = torch.stack(labels_list, dim=0)
        else:
            # Empty dataset
            if self.multi_label:
                self.labels = torch.empty((0, len(self.classes)), dtype=torch.float32)
            else:
                self.labels = torch.empty((0,), dtype=torch.long)

        print(f"Built labels tensor with shape: {self.labels.shape}")
        
    def __len__(self):
        return len(self.samples)
    
    def get_audio_for_position(self, patient_id, position):
        """Get one randomly selected audio segment for given patient and position"""
        segments = self.patient_segments[patient_id][position]
        if not segments:
            return None
            
        # Randomly select one segment file
        selected_file = random.choice(segments)
        
        try:
            wav, sr = librosa.load(selected_file, sr=self.cfg.sample_rate, mono=True)
            wav = torch.tensor(wav).to(torch.float32)
            assert sr == self.cfg.sample_rate, f'Invalid sampling rate: {sr} Hz, expected: {self.cfg.sample_rate} Hz.'
            return wav
        except Exception as e:
            logging.error(f"Error loading {selected_file}: {e}")
            return None
    
    def get_audio(self, index):
        """
        Override base method to return multi-position audio with mask.
        Returns: (5, unit_samples) tensor where:
        - First 4 channels: [MV, AV, PV, TV] audio data
        - 5th channel: mask information segmented across time dimension
        """
        patient_id, combo_id = self.samples[index]
        
        position_audios = []
        mask_values = []
        
        # Process each position
        for position in self.POSITIONS:
            wav = self.get_audio_for_position(patient_id, position)
            
            if wav is not None:
                # Apply length normalization
                l = len(wav)
                if l > self.unit_samples:
                    start = np.random.randint(l - self.unit_samples) if self.random_crop else 0
                    wav = wav[start:start + self.unit_samples]
                elif l < self.unit_samples:
                    wav = F.pad(wav, (0, self.unit_samples - l), mode='constant', value=0)
                mask_values.append(1.0)  # Available
            else:
                # Handle missing position
                if self.missing_strategy == 'zero_fill':
                    wav = torch.zeros(self.unit_samples)
                elif self.missing_strategy == 'mask_token':
                    wav = torch.full((self.unit_samples,), -999.0)  # Special value for missing
                else:
                    wav = torch.zeros(self.unit_samples)
                mask_values.append(0.0)  # Missing
            
            position_audios.append(wav)
        
        # ✅ Create mask channel with segmented mask information
        mask_channel = torch.zeros(self.unit_samples)
        segment_length = self.unit_samples // 4
        
        for i, mask_val in enumerate(mask_values):
            start_idx = i * segment_length
            # Handle the last segment to cover remaining samples
            end_idx = (i + 1) * segment_length if i < 3 else self.unit_samples
            mask_channel[start_idx:end_idx] = mask_val
        
        # Add mask channel to audio data
        position_audios.append(mask_channel)
        
        # Stack all channels: (5, unit_samples) - [MV, AV, PV, TV, mask]
        multi_wav = torch.stack(position_audios, dim=0)
        
        return multi_wav
    
    def get_label(self, index):
        """Get label for given index"""
        return self.labels[index]
    
    def __getitem__(self, index):
        """
        Returns (wav, label) format compatible with BaseRawAudioDataset
        wav shape: (5, unit_samples)
        """
        if self.return_filename:
            patient_id, combo_id = self.samples[index]
            filename = f"patient_{patient_id}_combo_{combo_id}"
            label = self.get_label(index)
            return filename if label is None else (filename, label)
        
        # Get multi-position audio with mask (already includes length normalization)
        wav = self.get_audio(index)  # Shape: (5, unit_samples)
        label = self.get_label(index)
        
        # Convert to float
        wav = wav.to(torch.float)
        
        # Apply transforms to audio channels (not mask channel)
        if self.tfms is not None:
            # Extract current mask to avoid transforming missing data
            mask_channel = wav[4]  # Save mask channel
            
            # Extract mask values for each position
            segment_length = self.unit_samples // 4
            position_masks = []
            for i in range(4):
                start_idx = i * segment_length
                mask_val = mask_channel[start_idx].item()  # Get mask value for this position
                position_masks.append(mask_val > 0)  # True if position has data
            
            # Apply transforms only to positions with real data
            for i in range(4):  # Only audio channels, not mask channel
                if position_masks[i]:  # Only transform real audio
                    wav[i] = self.tfms(wav[i])
        
        # Return (wav, label) format
        return wav if label is None else (wav, label)

    # ✅ 添加辅助方法来提取mask信息
    def extract_mask_from_audio(self, wav):
        """
        Extract mask array from the 5th channel of audio tensor
        Args:
            wav: (5, unit_samples) tensor from __getitem__
        Returns:
            mask: (4,) tensor with mask values for [MV, AV, PV, TV]
        """
        if wav.size(0) != 5:
            raise ValueError(f"Expected 5-channel audio, got {wav.size(0)} channels")
        
        mask_channel = wav[4]  # (unit_samples,)
        segment_length = len(mask_channel) // 4
        
        mask = torch.zeros(4)
        for i in range(4):
            start_idx = i * segment_length
            mask[i] = mask_channel[start_idx]  # Take first value of each segment
        
        return mask
    
    def extract_audio_from_wav(self, wav):
        """
        Extract audio data from the first 4 channels
        Args:
            wav: (5, unit_samples) tensor from __getitem__
        Returns:
            audio: (4, unit_samples) tensor with audio data for [MV, AV, PV, TV]
        """
        if wav.size(0) != 5:
            raise ValueError(f"Expected 5-channel audio, got {wav.size(0)} channels")
        
        return wav[:4]  # First 4 channels are audio data

def create_dataloader(cfg, fold=1, seed=42, batch_size=None, always_one_hot=False, balanced_random=False, pin_memory=True, num_workers=8):
    batch_size = batch_size or cfg.batch_size
    train_dataset = WavDataset_paired(cfg, 'train', holdout_fold=fold, always_one_hot=always_one_hot, random_crop=True)
    valid_dataset = WavDataset_paired(cfg, 'valid', holdout_fold=fold, always_one_hot=always_one_hot, random_crop=True,
        classes=train_dataset.classes)
    test_dataset = WavDataset_paired(cfg, 'test', holdout_fold=fold, always_one_hot=always_one_hot, random_crop=False,
        classes=train_dataset.classes)
    logging.info(f' classes: {train_dataset.classes}')

    if balanced_random:
        train_sampler = BalancedRandomSampler(train_dataset, batch_size, seed) if train_dataset.multi_label else \
            InfiniteSampler(train_dataset, batch_size, seed, shuffle=True)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_sampler, pin_memory=pin_memory,
                                            num_workers=num_workers) if balanced_random else \
                   torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory,
                                            num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory,
                                           num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory,
                                           num_workers=num_workers)

    return (train_loader, valid_loader, test_loader, train_dataset.multi_label)
