import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import spacy
from spacy.cli import download
from collections import Counter

class Multi30kDataset(Dataset):
    def __init__(self, split='train', src_vocab=None, tgt_vocab=None, max_len=100):
        """
        Loads the Multi30k dataset and prepares tokenizers.
        """
        self.split = split
        self.max_len = max_len
        
        # Load dataset from Hugging Face
        self.dataset = load_dataset("bentrevett/multi30k", split=self.split)
        
        # Load spacy tokenizers
        try:
            self.spacy_de = spacy.load("de_core_news_sm")
            self.spacy_en = spacy.load("en_core_web_sm")
        except OSError:
            print("Spacy models not found. Downloading them now...")
            download("de_core_news_sm")
            download("en_core_web_sm")
            self.spacy_de = spacy.load("de_core_news_sm")
            self.spacy_en = spacy.load("en_core_web_sm")

        self.unk_idx = 0
        self.pad_idx = 1
        self.sos_idx = 2
        self.eos_idx = 3
        self.special_tokens = ['<unk>', '<pad>', '<sos>', '<eos>']

        # If vocabs are not provided, we build them (typically only on train set)
        if src_vocab is None or tgt_vocab is None:
            self.src_vocab, self.tgt_vocab = self.build_vocab()
        else:
            self.src_vocab = src_vocab
            self.tgt_vocab = tgt_vocab

        # Pre-process all data to integers
        self.data = self.process_data()

    def tokenize_de(self, text):
        return [tok.text.lower() for tok in self.spacy_de.tokenizer(text)]

    def tokenize_en(self, text):
        return [tok.text.lower() for tok in self.spacy_en.tokenizer(text)]

    def build_vocab(self, min_freq=2):
        """
        Builds the vocabulary mapping for src (de) and tgt (en), including:
        <unk>, <pad>, <sos>, <eos>
        """
        src_counter = Counter()
        tgt_counter = Counter()

        print(f"Building vocabularies from {self.split} split...")
        for example in self.dataset:
            src_counter.update(self.tokenize_de(example['de']))
            tgt_counter.update(self.tokenize_en(example['en']))

        def create_vocab(counter):
            vocab = {tok: idx for idx, tok in enumerate(self.special_tokens)}
            idx = len(vocab)
            for word, freq in counter.items():
                if freq >= min_freq:
                    vocab[word] = idx
                    idx += 1
            return vocab

        src_vocab = create_vocab(src_counter)
        tgt_vocab = create_vocab(tgt_counter)
        return src_vocab, tgt_vocab

    def process_data(self):
        """
        Convert English and German sentences into integer token lists using
        spacy and the defined vocabulary. 
        """
        print(f"Processing data for {self.split} split...")
        processed_data = []
        for example in self.dataset:
            src_tokens = self.tokenize_de(example['de'])
            tgt_tokens = self.tokenize_en(example['en'])
            
            # Convert to indices
            src_indices = [self.sos_idx] + [self.src_vocab.get(tok, self.unk_idx) for tok in src_tokens] + [self.eos_idx]
            tgt_indices = [self.sos_idx] + [self.tgt_vocab.get(tok, self.unk_idx) for tok in tgt_tokens] + [self.eos_idx]
            
            # Truncate if necessary (optional depending on assignment rules, but safe for max_len)
            src_indices = src_indices[:self.max_len]
            tgt_indices = tgt_indices[:self.max_len]

            processed_data.append({
                'src': torch.tensor(src_indices, dtype=torch.long),
                'tgt': torch.tensor(tgt_indices, dtype=torch.long)
            })
        return processed_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]['src'], self.data[idx]['tgt']

def collate_fn(batch):
    """
    Collate function to pad batches to the maximum sequence length in the batch.
    """
    src_batch, tgt_batch = zip(*batch)
    
    # Pad sequences
    src_padded = torch.nn.utils.rnn.pad_sequence(src_batch, padding_value=1, batch_first=True) # pad_idx = 1
    tgt_padded = torch.nn.utils.rnn.pad_sequence(tgt_batch, padding_value=1, batch_first=True) # pad_idx = 1
    
    return src_padded, tgt_padded

def get_dataloaders(batch_size=32):
    """
    Helper function to get DataLoaders for train, val, and test splits.
    """
    train_dataset = Multi30kDataset(split='train')
    # Use training vocabularies for validation and testing
    val_dataset = Multi30kDataset(split='validation', src_vocab=train_dataset.src_vocab, tgt_vocab=train_dataset.tgt_vocab)
    test_dataset = Multi30kDataset(split='test', src_vocab=train_dataset.src_vocab, tgt_vocab=train_dataset.tgt_vocab)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader, train_dataset.src_vocab, train_dataset.tgt_vocab