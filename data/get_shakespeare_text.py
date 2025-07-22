import os
import re
import requests
import string
from collections import Counter
from typing import List, Dict, Tuple
import numpy as np
import pickle

class ShakespeareDataPrep:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.raw_text = ""
        self.cleaned_text = ""
        self.vocab = {}
        self.vocab_size = 0
        self.char_to_idx = {}
        self.idx_to_char = {}
        
        # Create data directory
        os.makedirs(data_dir, exist_ok=True)
    
    def download_shakespeare(self) -> str:
        """Download the complete works of Shakespeare from Project Gutenberg."""
        url = "https://www.gutenberg.org/files/100/100-0.txt"
        filepath = os.path.join(self.data_dir, "shakespeare_raw.txt")
        
        if os.path.exists(filepath):
            print(f"Shakespeare text already exists at {filepath}")
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        
        print("Downloading Shakespeare's complete works...")
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(response.text)
            
            print(f"Downloaded and saved to {filepath}")
            return response.text
            
        except Exception as e:
            print(f"Error downloading: {e}")
            return ""
    
    def clean_text(self, text: str) -> str:
        """Clean and preprocess the Shakespeare text."""
        print("Cleaning text...")
        
        # Remove Project Gutenberg header and footer
        # Look for the start of actual content
        start_markers = [
            "*** START OF THE PROJECT GUTENBERG EBOOK",
            "THE COMPLETE WORKS OF WILLIAM SHAKESPEARE"
        ]
        
        end_markers = [
            "*** END OF THE PROJECT GUTENBERG EBOOK",
            "End of the Project Gutenberg EBook"
        ]
        
        # Find start
        start_idx = 0
        for marker in start_markers:
            idx = text.find(marker)
            if idx != -1:
                # Find the end of this line and start from next line
                start_idx = text.find('\n', idx) + 1
                break
        
        # Find end
        end_idx = len(text)
        for marker in end_markers:
            idx = text.find(marker)
            if idx != -1:
                end_idx = idx
                break
        
        # Extract main content
        text = text[start_idx:end_idx]
        
        # Basic cleaning
        # Remove excessive whitespace but preserve paragraph structure
        text = re.sub(r'\n{3,}', '\n\n', text)  # Max 2 consecutive newlines
        text = re.sub(r'[ \t]+', ' ', text)     # Multiple spaces to single space
        text = re.sub(r' +\n', '\n', text)      # Remove trailing spaces
        text = re.sub(r'\n +', '\n', text)      # Remove leading spaces after newline
        
        # Remove chapter/scene markers that might interfere with learning
        # But keep act/scene structure as it's part of the literary format
        text = re.sub(r'^\s*SCENE [IVX]+\..*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*ACT [IVX]+\..*$', '', text, flags=re.MULTILINE)
        
        # Remove stage directions in brackets (optional - you might want to keep these)
        # text = re.sub(r'\[.*?\]', '', text)
        
        # Remove excessive punctuation repetition
        text = re.sub(r'[.]{3,}', '...', text)
        text = re.sub(r'[-]{2,}', '--', text)
        
        return text.strip()
    
    def build_vocabulary(self, text: str) -> Dict:
        """Build character-level vocabulary."""
        print("Building vocabulary...")
        
        # Get unique characters
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        
        # Create mappings
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(chars)}
        
        # Character frequency analysis
        char_freq = Counter(text)
        
        vocab_info = {
            'vocab_size': self.vocab_size,
            'char_to_idx': self.char_to_idx,
            'idx_to_char': self.idx_to_char,
            'char_frequencies': dict(char_freq.most_common(50)),
            'total_chars': len(text)
        }
        
        return vocab_info
    
    def encode_text(self, text: str) -> List[int]:
        """Encode text to integers using character vocabulary."""
        return [self.char_to_idx[ch] for ch in text]
    
    def decode_text(self, indices: List[int]) -> str:
        """Decode integers back to text."""
        return ''.join([self.idx_to_char[i] for i in indices])
    
    def create_train_val_split(self, text: str, val_split: float = 0.1) -> Tuple[str, str]:
        """Split text into train and validation sets."""
        split_idx = int(len(text) * (1 - val_split))
        train_text = text[:split_idx]
        val_text = text[split_idx:]
        
        return train_text, val_text
    
    def save_processed_data(self):
        """Save all processed data to files."""
        print("Saving processed data...")
        
        # Save raw and cleaned text
        with open(os.path.join(self.data_dir, "shakespeare_cleaned.txt"), 'w', encoding='utf-8') as f:
            f.write(self.cleaned_text)
        
        # Create train/val split
        train_text, val_text = self.create_train_val_split(self.cleaned_text)
        
        # Encode texts
        train_encoded = self.encode_text(train_text)
        val_encoded = self.encode_text(val_text)
        
        # Save encoded data as numpy arrays
        np.save(os.path.join(self.data_dir, "train_encoded.npy"), np.array(train_encoded, dtype=np.int32))
        np.save(os.path.join(self.data_dir, "val_encoded.npy"), np.array(val_encoded, dtype=np.int32))
        
        # Save vocabulary and metadata
        vocab_data = {
            'vocab_size': self.vocab_size,
            'char_to_idx': self.char_to_idx,
            'idx_to_char': self.idx_to_char,
            'train_size': len(train_encoded),
            'val_size': len(val_encoded)
        }
        
        with open(os.path.join(self.data_dir, "vocab.pkl"), 'wb') as f:
            pickle.dump(vocab_data, f)
        
        print(f"Data saved to {self.data_dir}/")
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Training samples: {len(train_encoded):,}")
        print(f"Validation samples: {len(val_encoded):,}")
    
    def analyze_dataset(self) -> Dict:
        """Analyze the dataset and return statistics."""
        if not self.cleaned_text:
            print("No cleaned text available. Run process_data() first.")
            return {}
        
        text = self.cleaned_text
        
        # Basic statistics
        total_chars = len(text)
        total_words = len(text.split())
        unique_chars = len(set(text))
        
        # Character frequency
        char_freq = Counter(text)
        most_common_chars = char_freq.most_common(10)
        
        # Line and paragraph stats
        lines = text.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        avg_line_length = np.mean([len(line) for line in non_empty_lines])
        
        # Word length distribution
        words = text.split()
        word_lengths = [len(word.strip(string.punctuation)) for word in words]
        avg_word_length = np.mean(word_lengths)
        
        stats = {
            'total_characters': total_chars,
            'total_words': total_words,
            'unique_characters': unique_chars,
            'vocabulary_size': self.vocab_size,
            'most_common_chars': most_common_chars,
            'average_line_length': avg_line_length,
            'average_word_length': avg_word_length,
            'total_lines': len(non_empty_lines)
        }
        
        return stats
    
    def process_data(self) -> Dict:
        """Main processing pipeline."""
        print("Starting Shakespeare dataset processing...")
        
        # Download data
        self.raw_text = self.download_shakespeare()
        if not self.raw_text:
            return {}
        
        # Clean text
        self.cleaned_text = self.clean_text(self.raw_text)
        
        # Build vocabulary
        vocab_info = self.build_vocabulary(self.cleaned_text)
        
        # Save processed data
        self.save_processed_data()
        
        # Analyze dataset
        stats = self.analyze_dataset()
        
        print("\n" + "="*50)
        print("DATASET ANALYSIS")
        print("="*50)
        print(f"Total characters: {stats['total_characters']:,}")
        print(f"Total words: {stats['total_words']:,}")
        print(f"Vocabulary size: {stats['vocabulary_size']}")
        print(f"Average line length: {stats['average_line_length']:.1f} chars")
        print(f"Average word length: {stats['average_word_length']:.1f} chars")
        print(f"\nMost common characters:")
        for char, freq in stats['most_common_chars']:
            char_display = repr(char) if char in ['\n', '\t', ' '] else char
            print(f"  {char_display}: {freq:,}")
        
        return stats

# Usage example
if __name__ == "__main__":
    # Initialize processor
    processor = ShakespeareDataPrep(data_dir="shakespeare_data")
    
    # Process the data
    stats = processor.process_data()
    
    # Example: Load processed data for training
    print("\n" + "="*50)
    print("LOADING PROCESSED DATA FOR TRAINING")
    print("="*50)
    
    # Load vocabulary
    with open("shakespeare_data/vocab.pkl", 'rb') as f:
        vocab_data = pickle.load(f)
    
    # Load encoded training data
    train_data = np.load("shakespeare_data/train_encoded.npy")
    val_data = np.load("shakespeare_data/val_encoded.npy")
    
    print(f"Loaded vocab size: {vocab_data['vocab_size']}")
    print(f"Train data shape: {train_data.shape}")
    print(f"Val data shape: {val_data.shape}")
    
    # Example: decode first 100 characters to verify
    first_100_chars = processor.decode_text(train_data[:100].tolist())
    print(f"\nFirst 100 characters of training data:")
    print(repr(first_100_chars[:200] + "..."))