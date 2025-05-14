import torch
import json
import os 
from utils import load_data, test_model, save_results, load_sentences, remove_eos, calculate_bleu, calculate_meteor,calculate_rouge 
from models import Seq2Seq, StyleExtractor
from transformers import BertTokenizer
import argparse

# Load configurations and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_models(config, model_save_path):
    # Initialize and load the models
    seq2seq = Seq2Seq(config).to(device)
    stex = StyleExtractor(config).to(device)

    seq2seq.load_state_dict(torch.load(os.path.join(model_save_path, "best_seq2seq.pkl")))
    stex.load_state_dict(torch.load(os.path.join(model_save_path, "best_style_extractor.pkl")))
    
    seq2seq.eval()
    stex.eval()
    seq2seq.decoder.mode = "infer"

    # Load the tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    return seq2seq, stex, tokenizer

def main(dataset, model_type):
    """
    Main function to run the model on a specified dataset and model type.

    Args:
        dataset (str): The dataset to use ('quora' or 'para').
        model_type (str): The model type to use ('org' or 'swa').
    """
    # Load data, model, and configuration
    data_folder = None
    config = None
    max_len = 15  # Maximum sequence length for inputs
    model_save_path = ""
    generated_file = None
    reference_file = None

    # Model-specific configurations (choose between "org" and "swa")
    if model_type == 'org':
        model_save_path = "save_model/"
    elif model_type == 'swa':
        model_save_path = "save_model_swa/"
    else:
        raise ValueError("Invalid model type. Choose between 'org' and 'swa'.")
    
    # Dataset-specific configurations
    if dataset == 'quora':
        model_save_path += "our_quora/"
        data_folder = 'datasets/processed/qqp-pos/data/'
        with open("quora_config.json", "r") as file:
            config = json.load(file)
        config["dataset"] = "quora"
        generated_file = "save_model/our_quora/trg_gen.txt"
        reference_file = 'dataset-text/quora-text/quora/test_trg.txt'
    elif dataset == 'para':
        model_save_path += "our_paranmt/"
        data_folder = 'datasets/processed/paranmt/data2/'
        with open("para_config.json", "r") as file:
            config = json.load(file)
        config["dataset"] = "para"  
        generated_file = "save_model/our_paranmt/trg_gen.txt"
        reference_file = 'dataset-text/para-text/para/test_trg.txt'

    
    # Load test data, models, and tokenizer
    test_loader, similarity_indices, idx_to_word, bert_outputs, reference_outputs = load_data(data_folder, max_len)
    seq2seq_model, style_extractor, tokenizer = load_models(config, model_save_path)

    try:
        # Run testing
        generated_responses, exemplar_indices = test_model(
            seq2seq_model, style_extractor, test_loader, similarity_indices, idx_to_word, bert_outputs, reference_outputs
        )

        # Save results
        save_results(
            generated_responses, exemplar_indices, model_save_path, idx_to_word, seq2seq_model, tokenizer, bert_outputs
        )

        # Load generated and reference sentences
        generated_sentences = load_sentences(generated_file)
        reference_sentences = load_sentences(reference_file)

        # Remove EOS tokens from generated sentences
        cleaned_generated_sentences = [remove_eos(sentence) for sentence in generated_sentences]

        # Calculate BLEU score
        bleu_score = calculate_bleu(reference_sentences, cleaned_generated_sentences)
        print(f"BLEU Score: {bleu_score:.4f}")

        # Calculate METEOR score
        avg_meteor_score = calculate_meteor(reference_sentences, cleaned_generated_sentences)
        print(f"METEOR Score: {avg_meteor_score:.4f}")

        # Calculate ROUGE scores
        avg_rouge1, avg_rouge2, avg_rougeL = calculate_rouge(reference_sentences, cleaned_generated_sentences)
        print(f"ROUGE-1 (R-1) Score: {avg_rouge1:.4f}")
        print(f"ROUGE-2 (R-2) Score: {avg_rouge2:.4f}")
        print(f"ROUGE-L (R-L) Score: {avg_rougeL:.4f}")

    finally:
        # Clear GPU memory and unload models
        print("Clearing GPU memory and unloading models...")
        del seq2seq_model
        del style_extractor
        torch.cuda.empty_cache()  # Release unused GPU memory
        print("GPU memory cleared.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the model on a specified dataset and model type.")
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
        help="Specify the model type to use: 'org' for original or 'swa' for Stochastic Weight Averaging."
    )
    args = parser.parse_args()

    # Run main with the dataset and model_type arguments
    main(args.dataset, args.model_type)
