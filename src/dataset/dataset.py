import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from collections import Counter

class Vocabulary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.word_count = Counter()
        self.index = 1  # Reserve 0 for padding

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.index
            self.idx2word[self.index] = word
            self.index += 1

    def build_vocab(self, captions):
        for caption in captions:
            self.word_count.update(caption.split())

        for word, count in self.word_count.items():
            if count >= 1:  # Include words with frequency >= 1
                self.add_word(word)

    def caption_to_seq(self, caption):
        return [self.word2idx[word] for word in caption.split() if word in self.word2idx]
    
    def load_from_dicts(self, w2i_dict, i2w_dict):
        # w2i_dict is word→index, i2w_dict is index→word (with string keys)
        self.word2idx = {word: int(idx) for word, idx in w2i_dict.items()}
        self.idx2word = {int(idx): word for idx, word in i2w_dict.items()}

    def seq_to_caption(self, seq):
        # Convert a list of indices back to a string, skipping any out‐of‐vocab indices
        words = []
        for idx in seq:
            if idx in self.idx2word:
                w = self.idx2word[idx]
                if w == "endseq" : 
                    break
                words.append(w)
        # You’ll likely see “startseq” and “endseq” tokens in front/back—feel free to strip them
        caption = " ".join(words)
        caption = caption.replace("startseq", "").replace("endseq", "").strip()
        return caption


class CaptionDataset(Dataset):
    def __init__(self, image_ids, mapp, features, vocab, max_length):
        self.image_ids = image_ids
        self.mapp = mapp
        self.features = features
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        captions = self.mapp[image_id]

        image_feature = torch.tensor(self.features[image_id], dtype=torch.float32)

        seqs = []
        for caption in captions:
            seq = self.vocab.caption_to_seq(caption)
            seqs.append(torch.tensor(seq, dtype=torch.long))

        padded_seqs = pad_sequence(seqs, batch_first=True, padding_value=0)

        if padded_seqs.size(1) < self.max_length:
            padded_seqs = torch.nn.functional.pad(padded_seqs, (0, self.max_length - padded_seqs.size(1)), value=0)
        else:
            padded_seqs = padded_seqs[:, :self.max_length]

        return image_feature, padded_seqs


def collate_fn(batch):
    features, sequences = zip(*batch)
    features = torch.stack(features)
    padded_sequences = pad_sequence([seq for seq in sequences], batch_first=True, padding_value=0)
    return features, padded_sequences


def create_dataloader(mapp, features, batch_size=16, max_length=50, test_split=0.1):
    image_ids = list(mapp.keys())
    train_ids, test_ids = train_test_split(image_ids, test_size=test_split)

    vocab = Vocabulary()
    all_captions = [caption for captions in mapp.values() for caption in captions]
    vocab.build_vocab(all_captions)

    train_dataset = CaptionDataset(train_ids, mapp, features, vocab, max_length)
    test_dataset = CaptionDataset(test_ids, mapp, features, vocab, max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)

    return train_loader, test_loader, vocab, test_ids
