import pickle
import numpy as np
import torch
import json
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu
from rouge_score import rouge_scorer
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os 
from data_preparation import TextDataset

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

glove_path = "glove/glove.6B.300d.txt"

def load_pickle(file_path):
    """
    Load data from a pickle file.

    Args:
        file_path (str): Path to the pickle file.

    Returns:
        object: Data loaded from the pickle file.
    """
    with open(file_path, "rb") as file:
        return pickle.load(file)


def save_pickle(file_path, data):
    """
    Save data to a pickle file.

    Args:
        file_path (str): Path to save the pickle file.
        data (object): Data to be saved.
    """
    with open(file_path, "wb") as file:
        pickle.dump(data, file)

def load_dataset_config(dataset_name):
    """
    Load dataset-specific configuration and paths.
    
    Args:
        dataset_name (str): The name of the dataset ('quora' or 'para').
        
    Returns:
        tuple: A tuple containing the dataset folder path and configuration dictionary.
        
    Raises:
        ValueError: If an unknown dataset name is provided.
    """
    base_model_path = "save_model/"
    dataset_config_path = ""

    if dataset_name == 'quora':
        dataset_folder = 'datasets/processed/qqp-pos/data/'
        dataset_config_path = "quora_config.json"
        base_model_path += "our_quora/"
    elif dataset_name == 'para':
        dataset_folder = 'datasets/processed/paranmt/data2/'
        dataset_config_path = "para_config.json"
        base_model_path += "our_paranmt/"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Load configuration from the specified JSON file
    with open(dataset_config_path, "r") as config_file:
        dataset_config = json.load(config_file)

    dataset_config["dataset"] = dataset_name  # Add dataset name to config
    return dataset_folder, base_model_path, dataset_config

def load_glove_embeddings(file_path):
    """
    Load GloVe embeddings into a dictionary.

    Args:
        file_path (str): Path to the GloVe embeddings file.

    Returns:
        dict: A dictionary mapping words to their embedding vectors.

    Raises:
        IOError: If the file cannot be read.
    """
    print(f"Loading GloVe embeddings from {file_path}...")
    glove_embeddings = {}

    with open(file_path, "r", encoding="utf-8") as glove_file:
        for line_number, line in enumerate(glove_file, 1):
            try:
                # Split line into word and embedding vector components
                components = line.split()
                word = components[0]
                embedding_vector = np.array(components[1:], dtype="float32")
                glove_embeddings[word] = embedding_vector
            except ValueError as e:
                # Log lines with issues but continue processing
                print(f"Skipping line {line_number}: {e}")

    print(f"Loaded {len(glove_embeddings)} word vectors.")
    return glove_embeddings


def load_glove_model(file_path):
    """
    Load the GloVe model into a dictionary.

    Args:
        file_path (str): Path to the GloVe embeddings file.

    Returns:
        dict: A dictionary mapping words to their embedding vectors.

    Raises:
        IOError: If the file cannot be read.
    """
    print(f"Loading GloVe model from {file_path}...")
    glove_model = {}

    with open(file_path, "r", encoding="utf-8") as glove_file:
        for line_number, line in enumerate(glove_file, 1):
            try:
                # Split line into word and embedding vector components
                components = line.split()
                word = components[0]
                embedding_vector = np.array(components[1:], dtype="float32")
                glove_model[word] = embedding_vector
            except ValueError as e:
                # Log lines with issues but continue processing
                print(f"Skipping line {line_number}: {e}")

    print(f"{len(glove_model)} words loaded successfully!")
    return glove_model

def initialize_word_embeddings(config, glove_file=glove_path):
    """
    Initialize word embeddings using GloVe for the vocabulary of the specified dataset.

    Args:
        config (dict): Configuration dictionary containing dataset and other parameters.
        glove_file (str): Path to the GloVe embeddings file.

    Returns:
        np.ndarray: A 2D numpy array of shape (vocab_size, embedding_dim) representing word embeddings.
    """
    # Determine the vocabulary file based on the dataset
    vocab_file = (
        "datasets/processed/paranmt/data2/word2idx.pkl"
        if config["dataset"] == "para"
        else "datasets/processed/qqp-pos/data/word2idx.pkl"
    )

    # Load the vocabulary
    print(f"Loading vocabulary from {vocab_file}...")
    with open(vocab_file, "rb") as f:
        word_to_index = pickle.load(f)

    # Load GloVe embeddings
    print(f"Loading GloVe embeddings from {glove_file}...")
    glove_embeddings = load_glove_model(glove_file)

    # Initialize the embedding matrix
    embedding_dim = 300  # Assuming GloVe embeddings have 300 dimensions
    vocab_size = len(word_to_index)
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    # Track missing words
    missing_word_count = 0

    # Populate the embedding matrix
    for word, index in word_to_index.items():
        if word in glove_embeddings:
            embedding_matrix[index] = glove_embeddings[word]
        else:
            missing_word_count += 1

    # Log the number of missing words
    print(f"{missing_word_count} words are not in the GloVe embedding.")

    return embedding_matrix


# Training function
def train_epoch(seq2seq_model, style_extractor, dataloader, optimizer, epoch, loss_criterion):
    """
    Train the models for one epoch.

    Args:
        seq2seq_model (torch.nn.Module): Seq2Seq model for sequence generation.
        style_extractor (torch.nn.Module): Model for extracting style embeddings.
        dataloader (DataLoader): DataLoader for training data.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        epoch (int): Current epoch number (for logging).
        loss_criterion (torch.nn.Module): Contrastive loss function for style and content embeddings.

    Returns:
        list: Average losses [NLL loss, Content loss, Style loss] for the epoch.
    """
    # Set models to training mode
    seq2seq_model.train()
    seq2seq_model.decoder.mode = "train"
    style_extractor.train()

    # Initialize loss trackers
    nll_loss_values = []
    content_loss_values = []
    style_loss_values = []

    # Iterate over batches
    for (
        source_seq, 
        input_lengths, 
        target_seq, 
        decoder_inputs, 
        output_lengths, 
        bert_source, 
        bert_target, 
        bert_similarity, 
        content_target, 
        content_lengths
    ) in tqdm(dataloader, desc=f"Training Epoch {epoch+1}", leave=False):
        # Move data to the device
        source_seq = source_seq.to(device)
        input_lengths = input_lengths.to(device)
        target_seq = target_seq.to(device)
        decoder_inputs = decoder_inputs.to(device)
        output_lengths = output_lengths.to(device)
        bert_source = bert_source.to(device)
        bert_target = bert_target.to(device)
        bert_similarity = bert_similarity.to(device)
        content_target = content_target.to(device)
        content_lengths = content_lengths.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Extract style embeddings
        style_emb_target = style_extractor(bert_target)
        style_emb_similarity = style_extractor(bert_similarity)

        # Forward pass through Seq2Seq model
        nll_loss, source_hidden = seq2seq_model(
            source_seq, 
            input_lengths, 
            style_emb_similarity, 
            response=target_seq, 
            decoder_input=decoder_inputs
        )

        # Content embedding contrastive loss
        content_embeddings = seq2seq_model.emb_layer(content_target)
        _, target_hidden = seq2seq_model.encoder_layer(content_embeddings, content_lengths)

        normalized_src_hidden = F.normalize(source_hidden, dim=1)
        normalized_trg_hidden = F.normalize(target_hidden, dim=1)

        content_contrastive_input = torch.cat(
            (normalized_src_hidden.unsqueeze(1), normalized_trg_hidden.unsqueeze(1)), 
            dim=1
        )
        content_loss = loss_criterion(content_contrastive_input)

        # Style embedding contrastive loss
        normalized_style_target = F.normalize(style_emb_target[:, -1, :], dim=1)
        normalized_style_similarity = F.normalize(style_emb_similarity[:, -1, :], dim=1)

        style_contrastive_input = torch.cat(
            (normalized_style_target.unsqueeze(1), normalized_style_similarity.unsqueeze(1)), 
            dim=1
        )
        style_loss = loss_criterion(style_contrastive_input)

        # Total loss
        total_loss = nll_loss + 0.1 * (content_loss + style_loss)

        # Backpropagation and optimizer step
        total_loss.backward()
        optimizer.step()

        # Log losses
        nll_loss_values.append(nll_loss.cpu().item())
        content_loss_values.append(content_loss.cpu().item())
        style_loss_values.append(style_loss.cpu().item())

    # Calculate average losses
    avg_nll_loss = np.mean(nll_loss_values)
    avg_content_loss = np.mean(content_loss_values)
    avg_style_loss = np.mean(style_loss_values)

    # Print loss summary for the epoch with training metrics indication
    print(
        f"\nTraining Metrics (Epoch: {epoch+1}) | "
        f"NLL Loss: {avg_nll_loss:.4f} | "
        f"Content Loss: {avg_content_loss:.4f} | "
        f"Style Loss: {avg_style_loss:.4f}"
    )

    return [avg_nll_loss, avg_content_loss, avg_style_loss]

def eval_epoch(seq2seq_model, style_extractor, dataloader, epoch):
    """
    Evaluate the models on a given dataset for one epoch.

    Args:
        seq2seq_model (torch.nn.Module): Seq2Seq model for sequence generation.
        style_extractor (torch.nn.Module): Model for extracting style embeddings.
        dataloader (DataLoader): DataLoader for evaluation data.
        epoch (int): Current epoch number (for logging).

    Returns:
        list: Average losses [NLL loss, Perplexity] for the epoch.
    """
    # Set models to evaluation mode
    seq2seq_model.eval()
    seq2seq_model.decoder.mode = "eval"
    style_extractor.eval()

    # Initialize lists to store losses
    perplexity_values = []
    nll_values = []

    # Disable gradient computation for evaluation
    with torch.no_grad():
        # Iterate over the evaluation batches
        for (
            source_seq, 
            input_lengths, 
            target_seq, 
            decoder_inputs, 
            output_lengths, 
            bert_source, 
            bert_target, 
            bert_similarity, 
            content_target, 
            content_lengths
        ) in tqdm(dataloader, desc=f"Evaluating Epoch {epoch+1}", leave=False):

            # Move data to the device
            source_seq = source_seq.to(device)
            input_lengths = input_lengths.to(device)
            target_seq = target_seq.to(device)
            decoder_inputs = decoder_inputs.to(device)
            output_lengths = output_lengths.to(device)
            bert_source = bert_source.to(device)
            bert_target = bert_target.to(device)
            bert_similarity = bert_similarity.to(device)
            content_target = content_target.to(device)
            content_lengths = content_lengths.to(device)

            # Extract style embeddings from the style extractor
            style_emb_similarity = style_extractor(bert_similarity)

            # Forward pass through Seq2Seq model
            total_output, _ = seq2seq_model(
                source_seq, 
                input_lengths, 
                style_emb_similarity, 
                response=target_seq, 
                decoder_input=decoder_inputs
            )

            # Extract perplexity and NLL from the model output
            perplexity, nll = total_output

            # Append values to the respective lists
            perplexity_values.append(perplexity)
            nll_values.append(nll.cpu().item())

        # Compute average perplexity and NLL over all batches
        perplexity_tensor = torch.cat(perplexity_values, dim=0)
        perplexity_mask = (perplexity_tensor < 200).float()
        avg_perplexity = (perplexity_tensor * perplexity_mask).sum() / perplexity_mask.sum()
        avg_perplexity = avg_perplexity.cpu().item()

        # Compute average NLL
        avg_nll = np.mean(nll_values)

        # Log the results for the current epoch
        print(
            f"\nEvaluation Metrics (Epoch: {epoch+1}) | "
            f"NLL Loss: {avg_nll:.4f} | Perplexity: {avg_perplexity:.4f}"
        )

    return [avg_nll, avg_perplexity]


def process_example(seq2seq_model, style_extractor, source_seq, input_lengths, target_seq, decoder_inputs, similarity_indices, bert_outputs, idx2word, example_idx, reference_outputs):
    """
    Perform inference for a single example using Seq2Seq and StyleExtractor models.

    Args:
        seq2seq_model (torch.nn.Module): Seq2Seq model for sequence generation.
        style_extractor (torch.nn.Module): Model for extracting style embeddings.
        source_seq (torch.Tensor): Source sequence input.
        input_lengths (torch.Tensor): Lengths of the source sequences.
        target_seq (torch.Tensor): Target sequence for reference.
        decoder_inputs (torch.Tensor): Inputs to the decoder.
        similarity_indices (list[list[int]]): Indices of similar examples for each input.
        bert_outputs (torch.Tensor): BERT representations for style similarity.
        idx2word (dict): Mapping from token indices to words.
        example_idx (int): Index of the current example.
        reference_outputs (list[list[int]]): Ground-truth outputs for comparison.

    Returns:
        tuple: 
            - Selected output sequence (torch.Tensor).
            - Corresponding similarity index (int).
    """
    # Initialize list to store generated sequences for similar examples
    candidate_sequences = []

    # Iterate over similarity indices for the current example
    for sim_idx in similarity_indices[example_idx]:
        # Prepare BERT similarity tensor
        bert_sim_tensor = torch.zeros(15 + 2, dtype=torch.long)
        bert_sim_data = bert_outputs[sim_idx]
        bert_sim_tensor[:min(len(bert_sim_data), 15 + 2)] = torch.tensor(
            bert_sim_data[:min(15 + 2, len(bert_sim_data))]
        )
        bert_sim_tensor = bert_sim_tensor.unsqueeze(0).to(device)

        # Extract style embedding using the StyleExtractor
        style_embedding = style_extractor(bert_sim_tensor)

        # Generate output sequence using the Seq2Seq model
        generated_ids, _ = seq2seq_model(
            source_seq, input_lengths, style_embedding, response=target_seq, decoder_input=decoder_inputs
        )
        candidate_sequences.append(generated_ids)

    # Find the best candidate based on maximum coverage with the reference output
    best_candidate_idx, max_coverage = -1, -1
    for candidate_idx, candidate_seq in enumerate(candidate_sequences):
        # Calculate the intersection between reference and candidate output
        intersection = len(set(reference_outputs[example_idx]) & set(candidate_seq[0].tolist()))
        coverage = intersection / len(set(reference_outputs[example_idx]))

        # Update the best candidate if coverage is improved
        if coverage > max_coverage:
            max_coverage = coverage
            best_candidate_idx = candidate_idx

    # Return the selected sequence and corresponding similarity index
    return candidate_sequences[best_candidate_idx], similarity_indices[example_idx][best_candidate_idx]

def test_model(seq2seq_model, style_extractor, test_data_loader, similarity_indices, idx_to_word, bert_outputs, reference_outputs):
    """
    Evaluate the model on the test dataset.

    Args:
        seq2seq_model (torch.nn.Module): Seq2Seq model for sequence generation.
        style_extractor (torch.nn.Module): Model for extracting style embeddings.
        test_data_loader (DataLoader): DataLoader for the test dataset.
        similarity_indices (list[list[int]]): Indices of similar examples for each input.
        idx_to_word (dict): Mapping from token indices to words.
        bert_outputs (torch.Tensor): BERT representations for style similarity.
        reference_outputs (list[list[int]]): Ground-truth outputs for comparison.

    Returns:
        tuple:
            - List of generated responses (list of torch.Tensor).
            - List of exemplar indices corresponding to the selected responses.
    """
    generated_responses = []
    exemplar_indices = []
    example_count = 0

    # Set models to evaluation mode
    seq2seq_model.eval()
    style_extractor.eval()

    # Process the test dataset
    with torch.no_grad():
        for (
            source_seq, input_lengths, target_seq, decoder_inputs, output_lengths,
            bert_src, bert_trg, bert_sim, content_trg, content_lengths
        ) in tqdm(test_data_loader, desc="Processing Test Data"):

            # Move data to the device
            source_seq, input_lengths, target_seq, decoder_inputs = (
                source_seq.to(device), input_lengths.to(device), target_seq.to(device), decoder_inputs.to(device)
            )
            bert_src, bert_trg, bert_sim, content_trg, content_lengths = (
                bert_src.to(device), bert_trg.to(device), bert_sim.to(device), content_trg.to(device), content_lengths.to(device)
            )

            # Process the current example to generate the best response and corresponding exemplar index
            best_response, best_exemplar = process_example(
                seq2seq_model, style_extractor, source_seq, input_lengths, target_seq,
                decoder_inputs, similarity_indices, bert_outputs, idx_to_word, example_count, reference_outputs
            )

            generated_responses.append(best_response)
            exemplar_indices.append(best_exemplar)
            example_count += 1

    return generated_responses, exemplar_indices


def save_results(generated_responses, exemplar_indices, save_path, idx_to_word, seq2seq_model, tokenizer, bert_outputs):
    """
    Save the generated responses and exemplar outputs to files.

    Args:
        generated_responses (list[torch.Tensor]): List of generated responses.
        exemplar_indices (list[int]): List of exemplar indices corresponding to the responses.
        save_path (str): Directory to save the results.
        idx_to_word (dict): Mapping from token indices to words.
        seq2seq_model (torch.nn.Module): Seq2Seq model used for decoding.
        tokenizer (Tokenizer): Tokenizer for decoding exemplar outputs.
        bert_outputs (torch.Tensor): BERT representations used for exemplar outputs.

    Saves:
        - "trg_gen.txt": File containing generated text outputs.
        - "exm.txt": File containing exemplar text outputs.
    """
    # Prepare file paths for saving
    generated_output_file = os.path.join(save_path, "trg_gen.txt")
    exemplar_output_file = os.path.join(save_path, "exm.txt")

    # Save generated text outputs
    with open(generated_output_file, 'w') as gen_file:
        for response in generated_responses:
            decoded_sentences = seq2seq_model.getword(idx_to_word, response)
            for sentence in decoded_sentences:
                gen_file.write(' '.join(sentence) + '\n')

    # Save exemplar text outputs
    with open(exemplar_output_file, 'w') as exm_file:
        for exemplar_idx in exemplar_indices:
            decoded_text = tokenizer.decode(bert_outputs[exemplar_idx][1:-2])  # Exclude special tokens
            exm_file.write(decoded_text + '\n')


def load_data(data_folder, max_len):
    # Load test data and other necessary files
    test_set = TextDataset("test", dataset_root=data_folder, max_length=max_len)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    with open(os.path.join(data_folder, 'test/sim.pkl'), 'rb') as f:
        sim = pickle.load(f)
    with open(os.path.join(data_folder, 'idx2word.pkl'), 'rb') as f:
        idx2word = pickle.load(f)
    with open(os.path.join(data_folder, 'test/bert_trg.pkl'), 'rb') as f:
        bert_output = pickle.load(f)
    with open(os.path.join(data_folder, 'test/trg.pkl'), 'rb') as f:
        normal_output = pickle.load(f)
    
    return test_loader, sim, idx2word, bert_output, normal_output

# Function to remove EOS token and everything after it
def remove_eos(sentence, eos_token="EOS"):
    """Removes 'EOS' and everything that follows from the sentence."""
    if eos_token in sentence:
        eos_index = sentence.index(eos_token)
        return sentence[:eos_index]  # Keep everything before 'EOS'
    return sentence

# Load references and generated sentences from files
def load_sentences(filename):
    """Loads sentences from a file, splitting each line into tokens."""
    with open(filename, 'r') as f:
        sentences = [line.strip().split() for line in f.readlines()]
    return sentences

# Calculate BLEU Score
def calculate_bleu(reference_sentences, generated_sentences):
    bleu_score = corpus_bleu([[ref] for ref in reference_sentences], generated_sentences)
    return bleu_score

# Simple METEOR score function based on F1
def simple_meteor_score(reference, hypothesis):
    """A simple METEOR score based on unigram precision, recall, and F1."""
    ref_tokens = set(reference.split())
    hyp_tokens = set(hypothesis.split())
    
    intersection = len(ref_tokens & hyp_tokens)
    precision = intersection / len(hyp_tokens) if hyp_tokens else 0
    recall = intersection / len(ref_tokens) if ref_tokens else 0
    
    if precision + recall == 0:
        f1_score = 0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score

# Calculate average METEOR score
def calculate_meteor(reference_sentences, generated_sentences):
    """Calculate the average METEOR score over a dataset."""
    meteor_scores = [
        simple_meteor_score(' '.join(ref), ' '.join(gen)) 
        for ref, gen in zip(reference_sentences, generated_sentences)
    ]
    avg_meteor_score = sum(meteor_scores) / len(meteor_scores)
    return avg_meteor_score

# Calculate ROUGE scores
def calculate_rouge(reference_sentences, generated_sentences):
    """Calculate ROUGE-1, ROUGE-2, and ROUGE-L scores."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge1_scores, rouge2_scores, rougeL_scores = [], [], []
    
    for ref, gen in zip(reference_sentences, generated_sentences):
        scores = scorer.score(' '.join(ref), ' '.join(gen))
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)
    
    avg_rouge1 = sum(rouge1_scores) / len(rouge1_scores)
    avg_rouge2 = sum(rouge2_scores) / len(rouge2_scores)
    avg_rougeL = sum(rougeL_scores) / len(rougeL_scores)
    
    return avg_rouge1, avg_rouge2, avg_rougeL
