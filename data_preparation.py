import os
import torch
from torch.utils.data import Dataset
import random
import pickle

class TextDataset(Dataset):
    """
    Custom Dataset for loading text and BERT-related data.
    """
    def __init__(self, dataset_type, dataset_root="/kaggle/input/your-dataset", max_length=15):
        """
        Initializes the TextDataset object.

        Args:
            dataset_type (str): Type of the dataset ('train', 'valid', or 'test').
            dataset_root (str): Path to the root directory of the dataset.
            max_length (int): Maximum sequence length for truncation or padding.
        """
        # Ensure dataset_type is valid
        assert dataset_type in ['train', 'valid', 'test'], "Invalid dataset type. Must be 'train', 'valid', or 'test'."

        self.dataset_type = dataset_type  # Dataset type (train/valid/test)
        self.max_length = max_length  # Maximum allowed sequence length

        # Load vocabulary and index mappings
        self.word_to_index = self._load_from_pickle(os.path.join(dataset_root, 'word2idx.pkl'))
        self.index_to_word = self._load_from_pickle(os.path.join(dataset_root, 'idx2word.pkl'))

        # Load sequences and related data for the specified dataset type
        self.source_sequences = self._load_from_pickle(os.path.join(dataset_root, dataset_type, 'src.pkl'))
        self.target_sequences = self._load_from_pickle(os.path.join(dataset_root, dataset_type, 'trg.pkl'))
        self.similarity_indices = self._load_from_pickle(os.path.join(dataset_root, dataset_type, 'sim.pkl'))
        self.bert_source_tokens = self._load_from_pickle(os.path.join(dataset_root, dataset_type, 'bert_src.pkl'))
        self.bert_target_tokens = self._load_from_pickle(os.path.join(dataset_root, dataset_type, 'bert_trg.pkl'))

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return len(self.source_sequences)

    def __getitem__(self, idx):
        """
        Retrieves a single sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: Contains source, target, and BERT-related tensors.
        """
        # Get the source and target sequences
        src_seq = self.source_sequences[idx]
        trg_seq = self.target_sequences[idx]

        # Process the source sequence into tensor format
        src_tensor, src_len = self._process_sequence(src_seq, self.max_length)

        # Prepare the target sequence for decoder input/output
        trg_tensor, dec_input_tensor, trg_len = self._prepare_target_sequence(trg_seq, self.max_length)

        # Process BERT-related sequences for source, target, and similarity
        bert_src_tensor = self._process_bert_sequence(self.bert_source_tokens[idx], self.max_length + 2)
        bert_trg_tensor = self._process_bert_sequence(self.bert_target_tokens[idx], self.max_length + 2)
        bert_sim_tensor = self._process_bert_sequence(
            self.bert_target_tokens[random.choice(self.similarity_indices[idx])], self.max_length + 2
        )

        # Prepare the content target for additional processing (if needed)
        content_trg_tensor, content_trg_len = self._process_sequence(trg_seq, self.max_length)

        # Return all processed tensors and lengths
        return (
            src_tensor, src_len, trg_tensor, dec_input_tensor, trg_len,
            bert_src_tensor, bert_trg_tensor, bert_sim_tensor,
            content_trg_tensor, content_trg_len,
        )

    def _load_from_pickle(self, path):
        """
        Loads data from a pickle file.

        Args:
            path (str): Path to the pickle file.

        Returns:
            Any: Loaded object from the pickle file.
        """
        with open(path, 'rb') as f:
            return pickle.load(f)

    def _process_sequence(self, sequence, max_length):
        """
        Truncates or pads a sequence to the desired maximum length.

        Args:
            sequence (list): Sequence of token indices.
            max_length (int): Maximum sequence length.

        Returns:
            torch.Tensor: Processed sequence tensor.
            int: Actual sequence length.
        """
        seq_len = len(sequence)  # Original sequence length
        tensor = torch.zeros(max_length, dtype=torch.long)  # Initialize tensor with zeros

        # Truncate or pad the sequence
        if seq_len > max_length:
            tensor[:max_length] = torch.tensor(sequence[:max_length])
            seq_len = max_length
        else:
            tensor[:seq_len] = torch.tensor(sequence)

        return tensor, seq_len

    def _prepare_target_sequence(self, sequence, max_length):
        """
        Prepares target sequence for the decoder with SOS and EOS tokens.

        Args:
            sequence (list): Target sequence of token indices.
            max_length (int): Maximum sequence length.

        Returns:
            tuple: Target tensor, decoder input tensor, and sequence length.
        """
        # Initialize tensors for the target and decoder input sequences
        target_tensor = torch.zeros(max_length + 1, dtype=torch.long)
        decoder_input_tensor = torch.zeros(max_length + 1, dtype=torch.long)

        seq_len = len(sequence)  # Original target sequence length

        # Truncate or pad the target sequence
        if seq_len > max_length:
            target_tensor[:max_length] = torch.tensor(sequence[:max_length])
            target_tensor[max_length] = 3  # Add EOS token
            decoder_input_tensor[1:max_length + 1] = torch.tensor(sequence[:max_length])
            decoder_input_tensor[0] = 2  # Add SOS token
            seq_len = max_length + 1
        else:
            target_tensor[:seq_len] = torch.tensor(sequence)
            target_tensor[seq_len] = 3  # Add EOS token
            decoder_input_tensor[1:seq_len + 1] = torch.tensor(sequence)
            decoder_input_tensor[0] = 2  # Add SOS token
            seq_len += 1

        return target_tensor, decoder_input_tensor, seq_len

    def _process_bert_sequence(self, bert_seq, max_length):
        """
        Truncates or pads a BERT token sequence to the desired maximum length.

        Args:
            bert_seq (list): Sequence of BERT token indices.
            max_length (int): Maximum sequence length.

        Returns:
            torch.Tensor: Processed BERT sequence tensor.
        """
        bert_tensor = torch.zeros(max_length, dtype=torch.long)  # Initialize tensor with zeros
        bert_tensor[:min(len(bert_seq), max_length)] = torch.tensor(bert_seq[:max_length])  # Truncate or pad
        return bert_tensor
