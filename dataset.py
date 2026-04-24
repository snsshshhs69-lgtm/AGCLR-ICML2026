

import os
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer
from transformers.data.data_collator import pad_without_fast_tokenizer_warning


# ══════════════════════════════════════════════════════════════════════════════
# DATASET CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

DATASET_CONFIG = {
    'gsm8k': {
        'hf_name': 'gsm8k',
        'hf_config': 'main',
        'answer_delimiter': '####',
        'max_reasoning_steps': 3,
        'has_explicit_steps': True,
    },
    'hotpotqa': {
        'hf_name': 'hotpotqa/hotpot_qa',
        'hf_config': 'fullwiki',
        'answer_delimiter': '###',
        'max_reasoning_steps': 4,
        'has_explicit_steps': False,  # No explicit reasoning steps
    },
    'prosqa': {
        'hf_name': None,  # Custom dataset
        'hf_config': None,
        'answer_delimiter': '###',
        'max_reasoning_steps': 6,
        'has_explicit_steps': True,
    }
}


# ══════════════════════════════════════════════════════════════════════════════
# DATASET LOADING FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def load_raw_dataset(dataset_name: str, data_dir: str = './data'):
    """
    Load raw dataset from Hugging Face or local files.
    
    Args:
        dataset_name: One of 'gsm8k', 'hotpotqa', 'prosqa'
        data_dir: Directory containing local dataset files
    
    Returns:
        Tuple of (train_data, val_data) as lists of dicts
    """
    config = DATASET_CONFIG[dataset_name]
    
    if config['hf_name']:
        # Load from Hugging Face
        print(f"📥 Loading {dataset_name} from Hugging Face...")
        
        try:
            if config['hf_config']:
                dataset = load_dataset(config['hf_name'], config['hf_config'])
            else:
                dataset = load_dataset(config['hf_name'])
            
            train_data = dataset['train']
            val_data = dataset['validation'] if 'validation' in dataset else dataset['test']
            
            # Convert to list of dicts
            train_data = [dict(item) for item in train_data]
            val_data = [dict(item) for item in val_data]
            
            print(f"✅ Loaded {len(train_data):,} train, {len(val_data):,} val samples")
            
        except Exception as e:
            print(f"❌ HuggingFace load failed: {e}")
            print(f"🔄 Trying local files in {data_dir}...")
            train_data, val_data = load_local_dataset(dataset_name, data_dir)
    
    else:
        # Load from local files (e.g., ProsQA)
        train_data, val_data = load_local_dataset(dataset_name, data_dir)
    
    # Add indices
    train_data = [{'idx': i, **item} for i, item in enumerate(train_data)]
    val_data = [{'idx': i, **item} for i, item in enumerate(val_data)]
    
    return train_data, val_data


def load_local_dataset(dataset_name: str, data_dir: str):
    """Load dataset from local JSON files"""
    train_file = Path(data_dir) / f"{dataset_name}_train.json"
    val_file = Path(data_dir) / f"{dataset_name}_val.json"
    
    if not train_file.exists():
        raise FileNotFoundError(
            f"Dataset files not found in {data_dir}. "
            f"Please download {dataset_name} dataset first."
        )
    
    with open(train_file) as f:
        train_data = json.load(f)
    with open(val_file) as f:
        val_data = json.load(f)
    
    print(f"✅ Loaded {len(train_data):,} train, {len(val_data):,} val samples from local files")
    
    return train_data, val_data


# ══════════════════════════════════════════════════════════════════════════════
# TOKENIZATION FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def tokenize_gsm8k_sample(sample, tokenizer):
    """Tokenize GSM8K sample with explicit reasoning steps"""
    # GSM8K format: question + reasoning steps + #### answer
    question = sample['question']
    
    # Extract reasoning steps and answer
    if 'answer' in sample:
        parts = sample['answer'].split('####')
        reasoning = parts[0].strip() if len(parts) > 0 else ''
        answer = parts[1].strip() if len(parts) > 1 else ''
        
        # Split reasoning into steps (by newlines or sentences)
        steps = [s.strip() for s in reasoning.split('\n') if s.strip()]
    else:
        steps = []
        answer = ''
    
    # Tokenize components
    question_tokenized = tokenizer.encode(question + '\n', add_special_tokens=True)
    steps_tokenized = [
        tokenizer.encode(step + '\n', add_special_tokens=False)
        for step in steps
    ]
    answer_tokenized = tokenizer.encode(
        '#### ' + answer, add_special_tokens=False
    ) + [tokenizer.eos_token_id]
    
    return {
        'question_tokenized': question_tokenized,
        'steps_tokenized': steps_tokenized,
        'answer_tokenized': answer_tokenized,
        'idx': sample['idx']
    }


def tokenize_hotpotqa_sample(sample, tokenizer):
    """
    Tokenize HotpotQA sample.
    Note: HotpotQA doesn't have explicit reasoning steps,
    so we use empty steps for curriculum compatibility.
    """
    question = sample['question']
    answer = sample['answer']
    
    # Tokenize
    question_tokenized = tokenizer.encode(question + '\n', add_special_tokens=True)
    steps_tokenized = [[]]  # Empty - no explicit steps in HotpotQA
    answer_tokenized = tokenizer.encode(
        '### ' + answer, add_special_tokens=False
    ) + [tokenizer.eos_token_id]
    
    return {
        'question_tokenized': question_tokenized,
        'steps_tokenized': steps_tokenized,
        'answer_tokenized': answer_tokenized,
        'idx': sample['idx'],
        'hop_count': sample.get('hop_count', 2)
    }


def tokenize_prosqa_sample(sample, tokenizer):
    """Tokenize ProsQA sample with explicit planning steps"""
    question = sample['question']
    steps = sample.get('reasoning_steps', [])
    answer = sample['answer']
    
    # Tokenize
    question_tokenized = tokenizer.encode(question + '\n', add_special_tokens=True)
    steps_tokenized = [
        tokenizer.encode(step + '\n', add_special_tokens=False)
        for step in steps
    ]
    answer_tokenized = tokenizer.encode(
        '### ' + answer, add_special_tokens=False
    ) + [tokenizer.eos_token_id]
    
    return {
        'question_tokenized': question_tokenized,
        'steps_tokenized': steps_tokenized,
        'answer_tokenized': answer_tokenized,
        'idx': sample['idx']
    }


TOKENIZE_FUNCTIONS = {
    'gsm8k': tokenize_gsm8k_sample,
    'hotpotqa': tokenize_hotpotqa_sample,
    'prosqa': tokenize_prosqa_sample
}


def get_base_dataset(dataset_name: str, data_list: List[Dict], tokenizer: AutoTokenizer):
    """
    Create base tokenized dataset.
    
    Args:
        dataset_name: Dataset identifier
        data_list: List of raw data samples
        tokenizer: Hugging Face tokenizer
    
    Returns:
        HuggingFace Dataset with tokenized samples
    """
    tokenize_fn = TOKENIZE_FUNCTIONS[dataset_name]
    
    # Create HF Dataset
    keys = data_list[0].keys()
    hf_dataset = Dataset.from_dict({k: [d[k] for d in data_list] for k in keys})
    
    # Tokenize
    print(f"🔤 Tokenizing {len(data_list):,} {dataset_name} samples...")
    tokenized_dataset = hf_dataset.map(
        lambda sample: tokenize_fn(sample, tokenizer),
        remove_columns=list(hf_dataset.features),
        num_proc=4,
        desc=f"Tokenizing {dataset_name}"
    )
    
    print(f"✅ Tokenization complete")
    return tokenized_dataset


# ══════════════════════════════════════════════════════════════════════════════
# CURRICULUM LEARNING DATASET
# ══════════════════════════════════════════════════════════════════════════════

def get_curriculum_dataset(
    stage: int,
    base_dataset: Dataset,
    num_latent_per_step: int,
    start_token_id: int,
    latent_token_id: int,
    end_token_id: int,
    max_stage: int = 3,
    uniform_prob: float = 0.0,
    shuffle: bool = False
):
    """
    Create curriculum dataset for a specific stage.
    
    Stages:
        0: Full CoT (no latent tokens)
        1-3: Progressive replacement of reasoning steps with latent tokens
    
    Args:
        stage: Current curriculum stage (0-3)
        base_dataset: Tokenized base dataset
        num_latent_per_step: Number of latent tokens per reasoning step (c_thought)
        start_token_id: <|start-latent|> token ID
        latent_token_id: <|latent|> token ID  
        end_token_id: <|end-latent|> token ID
        max_stage: Maximum curriculum stage
        uniform_prob: Probability of random stage selection
        shuffle: Whether to shuffle dataset
    
    Returns:
        Processed dataset with input_ids, labels, attention_mask
    """
    
    def process_sample(sample):
        # Random stage selection for regularization
        if random.random() < uniform_prob:
            current_stage = random.choice(list(range(max_stage + 1)))
        else:
            current_stage = stage
        
        # Calculate number of latent tokens
        num_steps_to_replace = min(current_stage, len(sample['steps_tokenized']))
        num_latent_tokens = num_steps_to_replace * num_latent_per_step
        
        # Build input sequence
        tokens = sample['question_tokenized'].copy()
        
        if num_latent_tokens > 0:
            # Add latent reasoning tokens
            tokens += [start_token_id]
            tokens += [latent_token_id] * num_latent_tokens
            tokens += [end_token_id]
        
        # Add remaining (non-latent) reasoning steps
        for step_tokens in sample['steps_tokenized'][num_steps_to_replace:]:
            tokens += step_tokens
        
        # Add answer
        tokens += sample['answer_tokenized']
        
        # Create labels (mask everything except answer)
        question_and_latent_len = (
            len(sample['question_tokenized']) + 
            (3 if num_latent_tokens > 0 else 0) +  # start + latents + end
            num_latent_tokens
        )
        
        labels = (
            [-100] * question_and_latent_len +
            tokens[question_and_latent_len:]
        )
        
        return {
            'input_ids': tokens,
            'labels': labels,
            'attention_mask': [1] * len(tokens),
            'position_ids': list(range(len(tokens))),
            'idx': sample['idx']
        }
    
    # Process dataset
    processed = base_dataset.map(
        process_sample,
        remove_columns=list(base_dataset.features),
        num_proc=4,
        desc=f"Processing Stage {stage}"
    )
    
    if shuffle:
        processed = processed.shuffle(seed=42)
    
    return processed


def get_evaluation_dataset(
    stage: int,
    base_dataset: Dataset,
    num_latent_per_step: int,
    start_token_id: int,
    latent_token_id: int,
    end_token_id: int,
    max_stage: int = 3
):
    """
    Create evaluation dataset (question + latent tokens only, no answer).
    
    Used for generating answers during evaluation.
    """
    
    def process_sample(sample):
        # Calculate latent tokens for this stage
        num_latent_tokens = min(stage, max_stage) * num_latent_per_step
        
        # Build sequence: question + latent tokens (no answer)
        tokens = sample['question_tokenized'].copy()
        
        if num_latent_tokens > 0:
            tokens += [start_token_id]
            tokens += [latent_token_id] * num_latent_tokens
            tokens += [end_token_id]
        
        return {
            'input_ids': tokens,
            'attention_mask': [1] * len(tokens),
            'position_ids': list(range(len(tokens))),
            'idx': sample['idx']
        }
    
    return base_dataset.map(
        process_sample,
        remove_columns=list(base_dataset.features),
        num_proc=4,
        desc=f"Processing Eval Stage {stage}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# DATA COLLATOR
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class LatentCollator:
    """
    Custom data collator for latent reasoning.
    
    Handles KV cache optimization by padding latent tokens to align across batch.
    Based on Meta's MyCollator implementation.
    """
    
    tokenizer: AutoTokenizer
    latent_token_id: int
    label_pad_token_id: int = -100
    
    def __call__(self, features, return_tensors='pt'):
        # Find earliest latent token position in batch
        earliest_latent = [
            feature['input_ids'].index(self.latent_token_id)
            for feature in features
            if self.latent_token_id in feature['input_ids']
        ]
        
        # Align latent tokens across batch for KV cache efficiency
        if len(earliest_latent) > 0:
            latest_earliest_latent = max(earliest_latent)
            
            for feature in features:
                if self.latent_token_id in feature['input_ids']:
                    n_tok_pad = latest_earliest_latent - feature['input_ids'].index(self.latent_token_id)
                else:
                    n_tok_pad = 0
                
                # Pad to align latent tokens
                feature['position_ids'] = [0] * n_tok_pad + feature['position_ids']
                feature['input_ids'] = [self.tokenizer.pad_token_id] * n_tok_pad + feature['input_ids']
                feature['attention_mask'] = [0] * n_tok_pad + feature['attention_mask']
                
                if 'labels' in feature:
                    feature['labels'] = [self.label_pad_token_id] * n_tok_pad + feature['labels']
        
        # Separate labels and position_ids
        label_name = 'label' if 'label' in features[0].keys() else 'labels'
        non_label_features = [
            {k: v for k, v in feature.items() if k != label_name and k != 'position_ids'}
            for feature in features
        ]
        
        # Pad sequences
        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            non_label_features,
            padding=True,
            return_tensors=return_tensors
        )
        
        # Handle labels
        labels = [feature[label_name] for feature in features] if label_name in features[0] else None
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            batch['labels'] = [
                label + [self.label_pad_token_id] * (max_label_length - len(label))
                for label in labels
            ]
            batch['labels'] = torch.tensor(batch['labels'], dtype=torch.int64)
        
        # Handle position_ids
        position_ids = [feature['position_ids'] for feature in features] if 'position_ids' in features[0] else None
        if position_ids is not None:
            max_pos_length = max(len(l) for l in position_ids)
            batch['position_ids'] = [
                pos_id + [0] * (max_pos_length - len(pos_id))
                for pos_id in position_ids
            ]
            batch['position_ids'] = torch.tensor(batch['position_ids'], dtype=torch.int64)
        
        return batch


# ══════════════════════════════════════════════════════════════════════════════
# HIGH-LEVEL API
# ══════════════════════════════════════════════════════════════════════════════

def get_dataset(dataset_name: str, data_dir: str = './data'):
    """
    Load and return train/val datasets.
    
    Returns:
        Tuple of (train_data, val_data) as lists of dicts
    """
    return load_raw_dataset(dataset_name, data_dir)


def get_dataloaders(
    dataset_name: str,
    tokenizer: AutoTokenizer,
    train_data: List[Dict],
    val_data: List[Dict],
    stage: int,
    num_latent_per_step: int,
    start_token_id: int,
    latent_token_id: int,
    end_token_id: int,
    batch_size: int = 16,
    shuffle_train: bool = True,
    num_workers: int = 4
):
    """
    Create train and validation dataloaders for a specific stage.
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create base datasets
    base_train = get_base_dataset(dataset_name, train_data, tokenizer)
    base_val = get_base_dataset(dataset_name, val_data, tokenizer)
    
    # Create curriculum datasets
    train_dataset = get_curriculum_dataset(
        stage, base_train, num_latent_per_step,
        start_token_id, latent_token_id, end_token_id,
        shuffle=shuffle_train
    )
    
    val_dataset = get_evaluation_dataset(
        stage, base_val, num_latent_per_step,
        start_token_id, latent_token_id, end_token_id
    )
    
    # Create collator
    collator = LatentCollator(
        tokenizer=tokenizer,
        latent_token_id=latent_token_id
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader
