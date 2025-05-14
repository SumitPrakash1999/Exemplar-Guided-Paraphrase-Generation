import os
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import json
from utils import train_epoch, eval_epoch, save_pickle
from models import Seq2Seq, StyleExtractor
from loss import SupConLoss
from data_preparation import TextDataset
import argparse

# Define the main function
def main(dataset):
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Training parameters
    batch_size = 128  # Number of samples per batch
    epochs = 1  # Total number of training epochs
    learning_rate = 1e-4  # Learning rate for the optimizer
    model_save_path = "save_model/"  # Directory to save model checkpoints
    max_len = 15  # Maximum sequence length for inputs
    data_folder = None
    config = None
    
    if dataset == 'quora':
        model_save_path += "our_quora/"
        data_folder = 'datasets/processed/qqp-pos/data/'
        # Load the configuration from the JSON file
        with open("quora_config.json", "r") as file:
            config = json.load(file)
        config["dataset"] = "quora" 
    elif dataset == 'para':
        model_save_path += "our_paranmt/"
        data_folder = 'datasets/processed/paranmt/data2/'
        # Load the configuration from the JSON file
        with open("para_config.json", "r") as file:
            config = json.load(file)
        config["dataset"] = "para"  

    # Initialize dataset loaders
    print("Initializing Datasets - - - ")
    train_dataset = TextDataset("train", dataset_root=data_folder, max_length=max_len)  # Training dataset
    valid_dataset = TextDataset("valid", dataset_root=data_folder, max_length=max_len)  # Validation dataset

    # Print the sizes of the datasets
    print(f"Size of Training Set: {len(train_dataset)}")
    print(f"Size of Validation Set: {len(valid_dataset)}")

    print("Creating  Dataloaders - - - ")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # DataLoader for training
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)  # DataLoader for validation


    # Initialize models
    print("Pushing models to device - - - ")
    seq2seq = Seq2Seq(config).to(device)  # Seq2Seq model
    style_extractor = StyleExtractor(config).to(device)  # Style extractor model

    # Define optimizer and loss function
    params = list(seq2seq.parameters()) + list(style_extractor.parameters())  # Parameters of both models
    optimizer = Adam(params, lr=learning_rate)  # Adam optimizer
    criterion = SupConLoss(temperature=0.5)  # Contrastive loss for style and content embeddings

    # Create directory for saving models if it doesn't exist
    os.makedirs(model_save_path, exist_ok=True)

    # Initialize variables to track the best model and lowest perplexity
    best_ppl = float('inf') 
    best_seq2seq_state = None
    best_stex_state = None
    loss_arr = []
    ppl_arr = []

    try:
        # Training loop
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")

            # Train the model for one epoch
            train_loss = train_epoch(seq2seq, style_extractor, train_loader, optimizer, epoch, criterion)
            nll, ppl = eval_epoch(seq2seq, style_extractor, valid_loader, epoch)

            # Append the losses and perplexity for logging purposes
            loss_arr.append(train_loss)
            ppl_arr.append([nll, ppl])
            
            # Check if the current PPL is the lowest
            current_ppl = ppl  
            if current_ppl < best_ppl:
                best_ppl = current_ppl
                best_seq2seq_state = seq2seq.state_dict()
                best_stex_state = style_extractor.state_dict()
                print(f"New best PPL found: {best_ppl:.4f} at epoch {epoch+1}")

        # After the loop, save the model with the best perplexity
        if best_seq2seq_state and best_stex_state:
            print(f"\nSaving best model with PPL: {best_ppl:.4f}")
            torch.save(best_seq2seq_state, os.path.join(model_save_path, 'best_seq2seq.pkl'))
            torch.save(best_stex_state, os.path.join(model_save_path, 'best_style_extractor.pkl'))
        else:
            print("\nSaving best model")
            torch.save(best_seq2seq_state, os.path.join(model_save_path, 'seq2seq.pkl'))
            torch.save(best_stex_state, os.path.join(model_save_path, 'style_extractor.pkl'))

        # Save loss and perplexity logs
        save_pickle(os.path.join(model_save_path, 'loss.pkl'), loss_arr)
        save_pickle(os.path.join(model_save_path, 'ppl.pkl'), ppl_arr)

    finally:
        # Clear GPU memory and delete model references
        print("\nCleaning up GPU memory and other resources.")
        del seq2seq
        del style_extractor
        torch.cuda.empty_cache()

        # Optionally clear other large variables
        del params, optimizer, criterion, best_seq2seq_state, best_stex_state, loss_arr, ppl_arr
        torch.cuda.empty_cache()
        print("Cleanup complete.")

# Run the main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the model on a specified dataset.")
    parser.add_argument(
        '--dataset', 
        type=str, 
        choices=['quora', 'para'], 
        required=True, 
        help="Specify the dataset to use: 'quora' or 'para'."
    )
    args = parser.parse_args()
    
    # Run main with the dataset argument
    main(args.dataset)
