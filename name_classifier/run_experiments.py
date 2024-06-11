from name_classifier.models.rnn import RNN
from name_classifier.data.data_setup import create_dataloaders
from name_classifier.data.utils import custom_collate_function
from name_classifier.engine import train
from name_classifier.utils import save_model
from name_classifier.utils import create_writer

import name_classifier.metadata.names as metadata

from pathlib import Path

import torch

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataloader, test_dataloader = create_dataloaders(
        root_dir = Path("data/names"),
        batch_size = 128,
        num_workers = 2,
        persistent_workers = True,
        pin_memory = True,
        collate_fn = custom_collate_function
    )

    num_epochs = [5, 7, 10]

    for epochs in num_epochs:
        model = RNN(metadata.N_LETTERS, 128, metadata.N_CATEGORIES)
        model_name = "Simple_RNN"
        model.to(device)
        
        print(f"[INFO] model name: {model_name}")
        print(f"[INFO] epochs: {epochs}")
        
        
        
        criterion = torch.nn.NLLLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr = 0.005)
        writer = create_writer(
            model_name,
            str(epochs)
        )
        
        train(
            model = model,
            train_dataloader = train_dataloader,
            test_dataloader = test_dataloader,
            criterion = criterion,
            optimizer = optimizer,
            epochs = epochs,
            device = device,
            writer = writer
        )
        
        file_path = f"simple_rnn_{epochs}_epochs.pth"
        
        save_model(
            model = model,
            dir = "models",
            model_name = file_path
        )
        
        print("-" * 100 + "\n")
    
if __name__ == "__main__":
    main()