"""A script to train a model."""

from pathlib import Path
import argparse
import torch

from data import data_setup
import name_classifier.metadata.names as metadata

from name_classifier.models.rnn import RNN
from name_classifier.models.lstm import LSTM
import name_classifier.utils as utils
import name_classifier.engine as engine

from torch.utils import tensorboard

from name_classifier.data.utils import custom_collate_function

HIDDEN_SIZE = 128


def _setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model", default="simple_rnn", type=str, help="a model to train"
    )

    parser.add_argument(
        "--learning_rate", type=float, default=0.005, help="learning rate"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="number of examples to load in a batch.",
    )

    parser.add_argument(
        "--epochs", type=int, default=5, help="number of epochs to train the model for."
    )

    return parser


def main():
    """Defines the script for training a model."""
    parser = _setup_parser()
    args = parser.parse_args()

    data_dir = Path("data/names")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")

    train_dataloader, test_dataloader = data_setup.create_dataloaders(
        root_dir=data_dir,
        batch_size=args.batch_size,
        collate_fn=custom_collate_function,
        num_workers=2,
        persistent_workers=True,
        pin_memory=True,
    )

    if args.model == "simple_rnn":
        model = RNN(metadata.N_LETTERS, 128, metadata.N_CATEGORIES)
    elif args.model == "simple_lstm":
        model = LSTM(metadata.N_LETTERS, 128, metadata.N_CATEGORIES)

    model.to(device)

    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    engine.train(
        model,
        train_dataloader,
        test_dataloader,
        criterion,
        optimizer,
        args.epochs,
        device,
        writer=tensorboard.SummaryWriter(),
    )

    utils.save_model(model, "models", model_name=args.model + ".pth")


if __name__ == "__main__":
    main()
