import torch
import torch.nn as nn
import numpy as np
from transformers import BertModel
from torch.optim import Adam
from utils import initialize_word_embeddings
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from loss import compute_nll_loss

TRANSFORMERS_CACHE='transformers-cache'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    """
    The Encoder uses a bidirectional GRU (Gated Recurrent Unit) to process the input sequence.
    It supports variable-length input sequences and outputs the sequence representations.

    Args:
        config_dict (dict): A dictionary containing the configuration parameters, including:
            - 'encoder': Dictionary containing encoder-specific parameters:
                - 'hidden_dim': The hidden state dimension of the GRU.
                - 'input_dim': The input feature dimension.
                - 'num_layers': Number of GRU layers.
                - 'bidirectional': Whether to use bidirectional GRU.
        drop_out (float): The dropout rate applied to the GRU layers.
        
    Attributes:
        rnn (nn.GRU): The GRU layer used for encoding the input sequence.

    """

    def __init__(self, config_dict):
        """
        Initializes the Encoder with the given configuration.

        Args:
            config_dict (dict): Configuration dictionary for setting hyperparameters like 
                                 input/output dimensions, number of layers, and bidirectional flag.
        """
        super(Encoder, self).__init__()

        # Extract hyperparameters from the config_dict
        encoder_config = config_dict['encoder']
        self.hidden_dim = encoder_config.get('hidden_dim', 256)
        self.input_dim = encoder_config.get('input_dim', 256)
        self.num_layers = encoder_config.get('num_layers', 1)
        self.bidirectional = bool(encoder_config.get('bidirectional', True))
        self.dropout = config_dict.get('drop_out', 0.2)

        # Initialize the GRU with the specified parameters
        self.rnn = nn.GRU(input_size=self.input_dim, 
                          hidden_size=self.hidden_dim, 
                          num_layers=self.num_layers, 
                          bias=True,
                          batch_first=True, 
                          dropout=(0 if self.num_layers == 1 else self.dropout), 
                          bidirectional=self.bidirectional)

    def forward(self, input_sequence, sequence_lengths):
        """
        Forward pass through the encoder.

        Args:
            input_sequence (torch.Tensor): The input sequence tensor of shape [batch_size, sequence_length, input_dim].
            sequence_lengths (torch.Tensor): A tensor containing the actual lengths of the input sequences in the batch.

        Returns:
            torch.Tensor: The encoded output sequence of shape [batch_size, sequence_length, hidden_dim * num_directions].
            torch.Tensor: The final hidden state of the encoder for each direction.
        """
        # Sort the sequence lengths in descending order (for packing)
        sorted_lengths, sorting_indices = sequence_lengths.sort(dim=-1, descending=True)
        sorted_input_sequence = input_sequence.index_select(0, sorting_indices)

        # Move sorted sequence lengths to CPU for pack_padded_sequence
        sorted_lengths_cpu = sorted_lengths.cpu()

        # Pack the padded input sequence (handles variable-length sequences)
        packed_input = pack_padded_sequence(sorted_input_sequence, sorted_lengths_cpu, batch_first=True)

        # Pass through the GRU
        packed_output, hidden_state = self.rnn(packed_input)

        # Unpack the sequence output to handle variable-length sequences
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        # Concatenate the hidden states from both directions (if bidirectional)
        hidden_state = torch.cat(tuple(hidden_state), dim=-1)

        # Revert the sorting of the sequences
        _, inverse_indices = sorting_indices.sort(dim=-1, descending=False)
        output = output.index_select(0, inverse_indices)
        hidden_state = hidden_state.index_select(0, inverse_indices)

        return output, hidden_state 


class ScaledDotAttention(nn.Module):
    """
    Implements Scaled Dot-Product Attention with optional bias and masking.

    Args:
        config_dict (dict): Configuration dictionary containing the model parameters.
            - 'decoder': Dictionary with 'hidden_dim' (query dimension).
            - 'encoder': Dictionary with 'final_out_dim' (key dimension).
        mode (str): Specifies the type of attention, 'Dot' for standard attention 
                    or 'Self' for self-attention where query and key dimensions are the same.

    Attributes:
        W (torch.nn.Parameter): Trainable weight matrix for projecting the query to key dimension.
    """

    def __init__(self, config_dict, mode='Dot'):
        super(ScaledDotAttention, self).__init__()

        # Set query and key dimensions
        query_dim = config_dict['decoder'].get('hidden_dim', 256)
        key_dim = config_dict['encoder'].get('final_out_dim', 256)

        if mode == 'Self':
            query_dim = key_dim  # Self-attention: query and key dimensions are equal

        # Initialize weight matrix
        self.weight_matrix = nn.Parameter(
            torch.empty(query_dim, key_dim, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")),
            requires_grad=True
        )
        nn.init.normal_(self.weight_matrix, mean=0., std=np.sqrt(2. / (query_dim + key_dim)))

    def forward(self, query, key, value, mask, bias=None):
        """
        Forward pass for scaled dot-product attention.

        Args:
            query (torch.Tensor): Query tensor of shape [batch_size, query_length, query_dim].
            key (torch.Tensor): Key tensor of shape [batch_size, key_length, key_dim].
            value (torch.Tensor): Value tensor of shape [batch_size, key_length, value_dim].
            mask (torch.Tensor): Mask tensor of shape [batch_size, key_length], used to mask out padded tokens.
            bias (torch.Tensor, optional): Bias tensor of shape [batch_size, key_length].

        Returns:
            torch.Tensor: Attention output of shape [batch_size, value_dim].
            torch.Tensor: Attention weights of shape [batch_size, key_length].
        """
        # Compute attention weights
        projected_query = query.matmul(self.weight_matrix)
        attention_scores = key.bmm(projected_query.unsqueeze(dim=2)).squeeze(dim=2)

        if bias is not None:
            attention_scores += bias

        # Apply mask to ignore padded positions
        mask = mask[:, :attention_scores.shape[-1]].bool()
        attention_scores.masked_fill_(mask, -float('inf'))

        # Compute attention probabilities
        attention_weights = torch.softmax(attention_scores, dim=-1)

        # Compute attention output
        attention_output = (attention_weights.unsqueeze(dim=2) * value).sum(dim=1)

        return attention_output, attention_weights


class GeneralAttention(nn.Module):
    """
    Implements a general attention mechanism with linear projections.

    Args:
        query_dim (int): Dimension of the query vector.
        key_dim (int): Dimension of the key vector.
        output_dim (int): Dimension of the final output vector.

    Attributes:
        attention_projection (torch.nn.Linear): Linear layer for projecting the query to match the key dimension.
        output_projection (torch.nn.Linear): Linear layer for mapping the output to the desired output dimension.
    """

    def __init__(self, query_dim, key_dim, output_dim):
        super(GeneralAttention, self).__init__()

        self.query_dim = query_dim
        self.key_dim = key_dim
        self.output_dim = output_dim

        # Define linear layers for projections
        self.attention_projection = nn.Linear(query_dim, key_dim)
        self.output_projection = nn.Linear(key_dim, output_dim)

    def forward(self, query, value):
        """
        Forward pass for general attention.

        Args:
            query (torch.Tensor): Query tensor of shape [batch_size, query_length, query_dim].
            value (torch.Tensor): Value tensor of shape [batch_size, value_length, value_dim].

        Returns:
            torch.Tensor: Attention output of shape [batch_size, output_dim].
        """
        # Project the query to match the key dimension
        query_projected = self.attention_projection(query).transpose(0, 1)

        # Compute attention weights
        attention_weights = torch.matmul(query_projected, value.transpose(1, 2)).transpose(1, 2)

        # Compute attention output
        attention_output = torch.matmul(value, attention_weights).squeeze(2)

        # Map attention output to the desired output dimension
        final_output = self.output_projection(attention_output)

        return final_output


class Decoder(nn.Module):
    """
    A Decoder module for sequence-to-sequence tasks. It incorporates 
    attention mechanisms, embeddings, and GRU layers for generating 
    sequences from encoded input and style features.

    Args:
        config_dict (dict): A dictionary containing configuration parameters 
                            for the decoder, encoder, style attention, and other components.

    Attributes:
        W_e2d (nn.Linear): A linear layer to transform the concatenated encoder hidden state 
                           and style feature into the decoder's initial hidden state.
        word_emb_layer (nn.Embedding): An embedding layer for converting input tokens 
                                       into dense vectors.
        attention_layer (ScaledDotAttention): A scaled dot-product attention mechanism 
                                              for attending over encoder outputs.
        gru (nn.GRU): A GRU layer for processing the combined context and input embeddings.
        projection_layer (nn.Linear): A linear layer to project the GRU outputs into the 
                                       vocabulary space for token predictions.
        style_attn (GeneralAttention): An attention mechanism for incorporating style embeddings.
        mode (str): Specifies the operational mode of the decoder. Can be 'train', 'eval', or 'infer'.
    """

    def __init__(self, config_dict):
        super().__init__()
        # Extract configuration parameters
        encoder_final_out_dim = config_dict['encoder'].get('final_out_dim', 256)
        decoder_hidden_dim = config_dict['encoder'].get('hidden_dim', 256)
        vocabulary_dim = config_dict.get('vocabulary_dim', 1)
        embedding_dim = config_dict.get('embedding_dim', 256)
        input_dim = config_dict['decoder'].get('input_dim', 256)
        hidden_dim = config_dict['decoder'].get('hidden_dim', 256)

        # Linear transformation for the encoder hidden state + style feature
        self.W_e2d = nn.Linear(encoder_final_out_dim + config_dict['style_attn']['style_in'], decoder_hidden_dim, bias=True)

        # Embedding layer for token inputs
        self.word_emb_layer = nn.Embedding(vocabulary_dim, embedding_dim)

        # Attention layer for encoder outputs
        self.attention_layer = ScaledDotAttention(config_dict)

        # GRU layer for sequence generation
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=1, bias=True,
                          batch_first=True, bidirectional=False)

        # Projection layer for GRU outputs to vocabulary logits
        self.projection_layer = nn.Linear(hidden_dim, vocabulary_dim)

        # General attention mechanism for style embeddings
        self.style_attn = GeneralAttention(
            config_dict['decoder'].get('hidden_dim', 256),
            config_dict['style_attn']['style_in'],
            config_dict['style_attn']['style_out']
        )

        # Decoder mode: 'train', 'eval', or 'infer'
        self.mode = 'train'

    def forward(self, encode_hidden, encode_output, encoder_mask, seq_label, decoder_input, style_emb, max_seq_len=21):
        """
        Forward pass of the decoder.

        Args:
            encode_hidden (torch.Tensor): Encoder's final hidden state (batch_size, encoder_hidden_dim).
            encode_output (torch.Tensor): Encoder outputs for attention (batch_size, seq_len, encoder_output_dim).
            encoder_mask (torch.Tensor): Mask for encoder outputs (batch_size, seq_len).
            seq_label (torch.Tensor): Ground-truth token sequence labels (batch_size, seq_len).
            decoder_input (torch.Tensor): Input tokens for the decoder (batch_size, seq_len).
            style_emb (torch.Tensor): Style embeddings for the sequence (batch_size, seq_len, style_dim).
            max_seq_len (int): Maximum sequence length for generation during inference. Default: 21.

        Returns:
            If mode is 'train':
                torch.Tensor: Negative log-likelihood (NLL) loss for the batch.
            If mode is 'eval':
                Tuple[torch.Tensor, torch.Tensor]: Perplexity and mean NLL loss for the batch.
            If mode is 'infer':
                torch.Tensor: Generated sequence IDs (batch_size, max_seq_len).
            Otherwise:
                None: Invalid mode.
        """
        # Extract style features and prepare the initial hidden state
        style_feature = style_emb[:, -1, :]  # Last style embedding
        encode_hidden = torch.cat([encode_hidden, style_feature], dim=-1)
        hidden = self.W_e2d(encode_hidden).unsqueeze(0)

        if self.mode in ['train', 'eval']:
            # Embedding the decoder inputs
            decoder_input_emb = self.word_emb_layer(decoder_input)
            decoder_output_arr = []

            # Process each time step sequentially
            for t in range(decoder_input.size(-1)):
                context, _ = self.attention_layer(hidden[-1], encode_output, encode_output, encoder_mask)
                output, hidden = self.gru(torch.cat([context, decoder_input_emb[:, t]], dim=-1).unsqueeze(1), hidden)
                decoder_output_arr.append(output.squeeze(1))

            # Project GRU outputs to vocabulary logits
            decoder_output = self.projection_layer(torch.stack(decoder_output_arr, dim=1))

            if self.mode == 'eval':
                loss = compute_nll_loss(decoder_output, seq_label, reduction='none')
                ppl = loss.exp()
                return ppl, loss.mean()
            else:
                loss = compute_nll_loss(decoder_output, seq_label, reduction='mean')
                return loss

        elif self.mode == 'infer':
            # Inference mode: generate sequences token-by-token
            id_arr = []
            previous_vec = self.word_emb_layer(
                torch.ones(size=[encode_output.size(0)], dtype=torch.long, device=device) * 2
            )  # Start token

            for t in range(max_seq_len):
                context, _ = self.attention_layer(hidden[-1], encode_output, encode_output, encoder_mask)
                output, hidden = self.gru(torch.cat([context, previous_vec], dim=-1).unsqueeze(1), hidden)
                decode_output = self.projection_layer(output.squeeze(1))
                _, previous_id = decode_output.max(dim=-1)
                previous_vec = self.word_emb_layer(previous_id)
                id_arr.append(previous_id)

            decoder_id = torch.stack(id_arr, dim=1)
            return decoder_id

        else:
            # Invalid mode
            return None

        
class Seq2Seq(nn.Module):
    """
    Sequence-to-Sequence Model combining an Encoder and a Decoder for sequence generation tasks.
    
    This model is designed to take in an input sequence, pass it through an encoder to generate hidden
    representations, and then decode it to generate the output sequence. It also incorporates style embeddings 
    to condition the generation process.

    Args:
        config_dict (dict): Configuration dictionary containing necessary hyperparameters such as:
            - 'embedding_dim': The dimensionality of the word embeddings.
            - 'vocabulary_dim': The size of the vocabulary.
            - 'lr': Learning rate for the optimizer.
        
    Attributes:
        emb_layer (nn.Embedding): The embedding layer for input sequences.
        encoder_layer (Encoder): The encoder network.
        decoder (Decoder): The decoder network for generating outputs.
        opt (Adam): The optimizer used to train the model.
    """
    
    def __init__(self, config_dict):
        """
        Initializes the Seq2Seq model, including embedding, encoder, decoder, and optimizer.
        
        Args:
            config_dict (dict): Configuration dictionary containing model parameters.
        """
        super(Seq2Seq, self).__init__()
        
        # Extract configurations for embedding and vocabulary dimensions
        embedding_dim = config_dict.get('embedding_dim', 1)
        vocabulary_dim = config_dict.get('vocabulary_dim', 1)
        
        # Initialize the embedding layer
        self.emb_layer = nn.Embedding(
            num_embeddings=vocabulary_dim, 
            embedding_dim=embedding_dim, 
            padding_idx=0  # Padding token index is set to 0
        )

        # Load pre-trained word embeddings (e.g., GloVe)
        glove_weight = initialize_word_embeddings(config_dict)
        self.emb_layer.weight.data.copy_(torch.from_numpy(glove_weight))

        # Initialize the encoder and decoder layers
        self.encoder_layer = Encoder(config_dict=config_dict)
        self.decoder = Decoder(config_dict)
        
        # Attach the embedding layer to the decoder for word embeddings
        self.decoder.word_emb_layer = self.emb_layer
        
        # Define parameters that require gradients
        self.trainable_params = list(filter(lambda p: p.requires_grad, self.parameters()))
        
        # Initialize the Adam optimizer
        self.optimizer = Adam(params=self.trainable_params, lr=config_dict.get('lr', 1e-4))

    def forward(self, seq_input, seq_len, style_emb, response=None, decoder_input=None, max_seq_len=16):
        """
        Forward pass through the Seq2Seq model. The input sequence is passed through the encoder and decoder.
        
        Args:
            seq_input (torch.Tensor): The input sequence of shape [batch_size, sequence_length].
            seq_len (torch.Tensor): Length of the input sequences for each batch item.
            style_emb (torch.Tensor): Style embeddings for conditioning the decoder output.
            response (torch.Tensor, optional): Ground truth sequence for teacher forcing during training.
            decoder_input (torch.Tensor, optional): Initial input for the decoder.
            max_seq_len (int, optional): The maximum sequence length for the output sequence. Defaults to 16.
        
        Returns:
            torch.Tensor: The output of the decoder (generated sequence).
            torch.Tensor: The final hidden state of the encoder.
        """
        # Create a mask for padding positions (0 is considered padding)
        encode_mask = (seq_input == 0).byte()
        
        # Get the word embeddings for the input sequence
        seq_input_embedded = self.emb_layer(seq_input)
        
        # Pass through the encoder
        encoder_output, encoder_hidden = self.encoder_layer(seq_input_embedded, seq_len)
        
        # Pass the encoder outputs and style embeddings to the decoder
        decoder_output = self.decoder(
            encoder_hidden, 
            encoder_output, 
            encode_mask, 
            response, 
            decoder_input, 
            style_emb, 
            max_seq_len=max_seq_len
        )
        
        return decoder_output, encoder_hidden

    def get_word_sequence(self, idx2word, batch_output):
        """
        Convert a sequence of word indices to actual words using a provided index-to-word mapping.
        
        Args:
            idx2word (dict): Mapping from index to word.
            batch_output (torch.Tensor): Tensor containing sequences of word indices.
        
        Returns:
            list: A list of tokenized sentences represented as words.
        """
        return [
            [idx2word[idx.item()] for idx in sentence if idx.item() != 0] 
            for sentence in batch_output
        ]


class StyleExtractor(nn.Module):
    """
    StyleExtractor class for extracting style embeddings from text using BERT.

    This module uses a pre-trained BERT model (`bert-base-uncased`) to obtain
    hidden states and then extracts the style embedding (first token representation).

    Args:
        config_dict (dict): Configuration dictionary containing any necessary configurations.
        
    Methods:
        forward(input_ids):
            Extracts style embeddings from input text using BERT.
            
    Returns:
        torch.Tensor: Style embeddings for the input text.
    """
    
    def __init__(self, config_dict):
        """
        Initializes the StyleExtractor with a pre-trained BERT model.

        Args:
            config_dict (dict): Configuration dictionary, currently unused but can be extended.
        """
        super(StyleExtractor, self).__init__()
        
        # Load the pre-trained BERT model from HuggingFace Transformers
        self.bert_model = BertModel.from_pretrained(
            "bert-base-uncased",  # Pre-trained BERT model
            output_hidden_states=True,  # Ensure hidden states are returned
            cache_dir=TRANSFORMERS_CACHE  # Specify cache directory for model storage
        )

    def forward(self, input_ids):
        """
        Extracts style embeddings from the input using the BERT model.

        Args:
            input_ids (torch.Tensor): Input token ids of shape [batch_size, sequence_length].

        Returns:
            torch.Tensor: The style embeddings, shaped as [batch_size, sequence_length, embedding_dim].
                          The style embeddings are extracted from the first hidden state token (CLS token).
        """
        # Forward pass through BERT
        outputs = self.bert_model(input_ids)
        
        # Extract all hidden states
        hidden_states = torch.stack(outputs.hidden_states, dim=1)
        
        # Extract the first token's hidden state (CLS token) from all layers
        cls_hidden_states = hidden_states[:, :, 0, :]  # Shape: [batch_size, num_layers, embedding_dim]
        
        # Return the hidden states corresponding to the CLS token, which represents the style embedding
        return cls_hidden_states
