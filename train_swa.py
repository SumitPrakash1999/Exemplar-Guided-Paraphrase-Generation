import os
import torch
import json
from models import Seq2Seq, StyleExtractor
from utils import train_epoch, eval_epoch, save_pickle
from data_preparation import TextDataset
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.optim.swa_utils as swa_utils  
from loss import SupConLoss

def main(dataset):
    """
    Main function to train and evaluate models with Stochastic Weight Averaging (SWA).
    """
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Training parameters
    batch_size = 128
    epochs = 10  # Total number of epochs
    learning_rate = 1e-4
    save_freq = 2  # Save model checkpoints every 'save_freq' epochs
    max_seq_length = 15
    model_save_path = "save_model_swa/"  # Directory to save model checkpoints
    swa_start_epoch = int(0.7 * epochs)  # SWA begins at 70% of total epochs
    swa_update_count = 0
    data_folder = None
    config = None

    # Load dataset-specific configurations
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

    # Initialize datasets and data loaders
    print("Initializing datasets...")
    train_dataset = TextDataset("train", dataset_root=data_folder, max_length=max_seq_length)
    valid_dataset = TextDataset("valid", dataset_root=data_folder, max_length=max_seq_length)
    print(f"Training set size: {len(train_dataset)}, Validation set size: {len(valid_dataset)}")

    print("Creating dataloaders...")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    # Initialize models
    print("Initializing models...")
    seq2seq = Seq2Seq(config).to(device)
    style_extractor = StyleExtractor(config).to(device)

    # SWA models
    swa_seq2seq = Seq2Seq(config).to(device)
    swa_stex = StyleExtractor(config).to(device)

    # Optimizer and loss function
    params = list(seq2seq.parameters()) + list(style_extractor.parameters())
    optimizer = Adam(params, lr=learning_rate)
    criterion = SupConLoss(temperature=0.5)

    # Tracking variables
    best_ppl = float('inf')  # Best perplexity
    best_seq2seq_state = None
    best_stex_state = None
    loss_arr, ppl_arr, swa_ppl_arr = [], [], []

    # Training loop
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        # Train and evaluate the models
        train_loss = train_epoch(seq2seq, style_extractor, train_loader, optimizer, epoch, criterion)
        nll, ppl = eval_epoch(seq2seq, style_extractor, valid_loader, epoch)

        # Log metrics
        loss_arr.append(train_loss)
        ppl_arr.append(ppl)
        print(f"Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, Validation PPL: {ppl:.4f}")

        # Check for best model
        if ppl < best_ppl:
            best_ppl = ppl
            best_seq2seq_state = seq2seq.state_dict()
            best_stex_state = style_extractor.state_dict()
            print(f"New best PPL: {best_ppl:.4f}")

        # Apply SWA starting from a specific epoch
        if epoch >= swa_start_epoch:
            print(f"Updating SWA models at epoch {epoch + 1}...")
            swa_update_count += 1

            # Update SWA model weights
            with torch.no_grad():
                for param, swa_param in zip(seq2seq.parameters(), swa_seq2seq.parameters()):
                    swa_param.data *= (swa_update_count - 1) / swa_update_count
                    swa_param.data += param.data / swa_update_count

                for param, swa_param in zip(style_extractor.parameters(), swa_stex.parameters()):
                    swa_param.data *= (swa_update_count - 1) / swa_update_count
                    swa_param.data += param.data / swa_update_count

            # Evaluate SWA models
            swa_utils.update_bn(train_loader, swa_seq2seq)  # Update batch norm statistics
            swa_utils.update_bn(train_loader, swa_stex)
            swa_nll, swa_ppl = eval_epoch(swa_seq2seq, swa_stex, valid_loader, epoch)
            swa_ppl_arr.append(swa_ppl)
            print(f"SWA Validation PPL: {swa_ppl:.4f}")

        # Save periodic checkpoints
        if (epoch + 1) % save_freq == 0:
            torch.save(seq2seq.state_dict(), os.path.join(model_save_path, f'seq2seq_epoch_{epoch + 1}.pkl'))
            torch.save(style_extractor.state_dict(), os.path.join(model_save_path, f'stex_epoch_{epoch + 1}.pkl'))

    # Save the best models
    print(f"\nSaving best models with PPL: {best_ppl:.4f}")
    torch.save(best_seq2seq_state, os.path.join(model_save_path, 'best_seq2seq.pkl'))
    torch.save(best_stex_state, os.path.join(model_save_path, 'best_style_extractor.pkl'))

    # Save SWA models
    print("Saving SWA models...")
    torch.save(swa_seq2seq.state_dict(), os.path.join(model_save_path, 'swa_seq2seq.pkl'))
    torch.save(swa_stex.state_dict(), os.path.join(model_save_path, 'swa_stex.pkl'))

    # Save metrics
    print("Saving logs...")
    save_pickle(os.path.join(model_save_path, 'loss.pkl'), loss_arr)
    save_pickle(os.path.join(model_save_path, 'ppl.pkl'), ppl_arr)
    save_pickle(os.path.join(model_save_path, 'swa_ppl.pkl'), swa_ppl_arr)

    # Clean up GPU memory
    print("Cleaning up GPU memory...")
    del seq2seq, style_extractor, swa_seq2seq, swa_stex
    torch.cuda.empty_cache()
    print("Cleanup complete.")

# Entry point
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train models with SWA on a specified dataset.")
    parser.add_argument('--dataset', type=str, choices=['quora', 'para'], required=True,
                        help="Specify the dataset to use: 'quora' or 'para'.")
    args = parser.parse_args()

    main(args.dataset)
