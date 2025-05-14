import pickle
import torch
from models import Seq2Seq, StyleExtractor
from transformers import BertTokenizer
from utils import load_dataset_config
import warnings
import os
import sys
import nltk
from tqdm import tqdm
import argparse

# Ignore warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_models(config, model_save_path, seq2seq_model_name, style_extractor_model_name):
    # Initialize and load the models
    seq2seq = Seq2Seq(config).to(device)
    stex = StyleExtractor(config).to(device)

    seq2seq.load_state_dict(torch.load(os.path.join(model_save_path, seq2seq_model_name)))
    stex.load_state_dict(torch.load(os.path.join(model_save_path, style_extractor_model_name)))
    
    seq2seq.eval()
    stex.eval()
    seq2seq.decoder.mode = "infer"

    # Load the tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    return seq2seq, stex, tokenizer

def create_packed_similar_tensor(similarity_data, max_length):
    """
    Create a packed similarity tensor for the input sequence.

    Args:
        similarity_data (list): A list of similarity values.
        max_length (int): The maximum allowed sequence length.

    Returns:
        torch.Tensor: A tensor containing the similarity data, padded or truncated to max_length.
    """
    packed_similarity = torch.zeros(max_length + 2, dtype=torch.long)
    packed_similarity[:min(len(similarity_data), max_length + 2)] = torch.tensor(
        similarity_data[:min(max_length + 2, len(similarity_data))]
    )
    return packed_similarity

def prepare_input_sequence(input_text, word_to_index, max_length):
    """
    Prepare an input sequence for the model, including tokenization, indexing, and padding.

    Args:
        input_text (list of str): The input text as a list of sentences.
        word_to_index (dict): A mapping from words to their corresponding indices.
        max_length (int): The maximum allowed sequence length.

    Returns:
        tuple: A tuple containing the padded input tensor and its length.
    """
    # Tokenize the input sentences
    tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in input_text]
    
    # Convert tokens to indices using the provided vocabulary
    indexed_sentences = [
        [word_to_index.get(word, word_to_index.get('UNK')) for word in tokens]
        for tokens in tokenized_sentences
    ]
    indexed_sentence = indexed_sentences[0]
    sentence_length = len(indexed_sentence)

    # Create a tensor for the input sequence with padding or truncation
    padded_input = torch.zeros(max_length, dtype=torch.long)
    if sentence_length > max_length:
        padded_input[:max_length] = torch.tensor(indexed_sentence[:max_length])
        sentence_length = max_length
    else:
        padded_input[:sentence_length] = torch.tensor(indexed_sentence)

    # Add batch dimension and create a tensor for the length
    padded_input = padded_input.unsqueeze(0)
    sequence_length = torch.tensor([sentence_length])

    return padded_input, sequence_length

def main(dataset_name, model_type):
    """
    Main function to perform paraphrase generation using the Seq2Seq and StyleExtractor models.
    
    Steps:
        1. Load dataset-specific configurations and paths.
        2. Initialize and load model weights.
        3. Prepare input data and perform inference for paraphrase generation.
        4. Save the generated output to a text file.
        5. Clear GPU memory after execution.
    """

    # Prompt user for input text
    input_text = input("Enter a sentence to generate paraphrases (max length = 15): ")

    # Check if input is empty
    if not input_text.strip():  # .strip() removes leading/trailing whitespace
        print("Error: The input text is empty. Please enter a valid sentence.")
        sys.exit()  # Stop the program

    # Configuration
    max_seq_length = 15
    data_folder = None
    config = None
    model_save_path = ""

    # Load dataset-specific configurations and paths
    data_folder, model_save_path, config, seq2seq_model_name, style_extractor_model_name = load_dataset_config(dataset_name, model_type)
    
    # Load vocabulary and test data
    with open(os.path.join(data_folder, 'idx2word.pkl'), 'rb') as f:
        idx_to_word = pickle.load(f)
    with open(os.path.join(data_folder, 'word2idx.pkl'), 'rb') as f:
        word_to_idx = pickle.load(f)
    with open(os.path.join(data_folder, 'test/bert_src.pkl'), 'rb') as f:
        style_representations = pickle.load(f)

    # Prepare input sequence for the model
    input_sequence, sequence_length = prepare_input_sequence(input_text, word_to_idx, max_seq_length)
    
    # Check if sequence length exceeds the maximum allowed length
    if sequence_length > max_seq_length:
        print(f"Error: The input sequence length is {sequence_length}, which exceeds the maximum allowed length of {max_seq_length}.")
        sys.exit()  # Stop the program if the sequence is too long

    input_sequence = input_sequence.to(device)
    sequence_length = sequence_length.to(device)

    # Load test data, models, and tokenizer
    seq2seq_model, style_extractor_model, _ = load_models(config, model_save_path, seq2seq_model_name, style_extractor_model_name)

    seq2seq_model.decoder.mode = "infer"

    # Perform inference to generate paraphrases
    generated_sequences = []
    with torch.no_grad():
        for style_vector in tqdm(style_representations, desc="Processing Style Representations"):
            packed_style_tensor = create_packed_similar_tensor(style_vector, max_seq_length)
            packed_style_tensor = packed_style_tensor.unsqueeze(0).to(device)

            # Extract style embeddings and generate paraphrases
            style_embedding = style_extractor_model(packed_style_tensor)
            predicted_ids, _ = seq2seq_model(input_sequence, sequence_length, style_embedding)
            generated_sequences.append(predicted_ids)

    # Save generated paraphrases to a file
    output_filename = os.path.join(model_save_path, "generated_paraphrases.txt")
    if os.path.exists(output_filename):
        os.remove(output_filename)
    with open(output_filename, 'a') as output_file:
        for batch in generated_sequences:
            paraphrases = seq2seq_model.get_word_sequence(idx_to_word, batch)  
            for paraphrase in paraphrases:
                output_file.write(' '.join(paraphrase))
                output_file.write('\n')

    # Clear GPU memory
    del seq2seq_model, style_extractor_model, input_sequence, style_representations
    torch.cuda.empty_cache()

    print(f"Paraphrases successfully generated and saved to {output_filename}.")

# Run the main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the paraphrase generation model on a specified dataset.")
    parser.add_argument(
        '--dataset', 
        type=str, 
        choices=['quora', 'para'], 
        required=True, 
        help="Specify the dataset to use: 'quora' or 'para'."
    )
    parser.add_argument(
        '--model_type',
        type=str,
        choices=['org', 'swa'],
        required=True,
        help="Specify the model type to use: 'org' for original model or 'swa' for Stochastic Weight Averaging model."
    )
    args = parser.parse_args()
    
    # Run main with the dataset and model_type arguments
    main(args.dataset, args.model_type)