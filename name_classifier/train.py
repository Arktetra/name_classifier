from pathlib import Path
import argparse
import torch

import data.data_setup as data_setup
import metadata.names as metadata

from models.rnn import RNN
import utils, engine

HIDDEN_SIZE = 128

def _setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--model",
        default = "simple_rnn",
        type = str,
        help = "a model to train"
    )
    
    parser.add_argument(
        "--learning_rate",
        type = float,
        default = 0.005,
        help = "learning rate"
    )
    
    parser.add_argument(
        "--batch_size",
        type = int,
        default = 1,
        help = "number of examples to load in a batch."
    )
    
    parser.add_argument(
        "--epochs",
        type = int,
        default = 5,
        help = "number of epochs to train the model for."
    )
    
    return parser

def main():
    parser = _setup_parser()
    args = parser.parse_args()
    
    
    
    data_dir = Path("data/names")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    train_dataloader, test_dataloader = data_setup.create_dataloaders(
        root_dir = data_dir,
        batch_size = args.batch_size,
        num_workers = 0,
        persistent_workers = False,
        pin_memory = False
    )

    model = RNN(metadata.N_LETTERS, 128, metadata.N_CATEGORIES)
    model.to(device)

    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = args.learning_rate)

    engine.train(
        model,
        train_dataloader,
        test_dataloader,
        criterion,
        optimizer,
        args.epochs,
        device
    )

    utils.save_model(
        model,
        "models",
        model_name = "simple_rnn.pth"
    )

if __name__ == "__main__":
    main()