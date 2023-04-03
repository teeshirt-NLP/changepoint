from model import CAPEmodel
from train import Trainer
from config import RUNTIME_SETTINGS, HYPERPARAMS

from data_loading import DataLoader

def main():    
    # Load the data
    loaded_data = DataLoader(RUNTIME_SETTINGS)

    # Initialize the model
    model = CAPEmodel(loaded_data, HYPERPARAMS)

    # Initialize the trainer
    trainer = Trainer(model, RUNTIME_SETTINGS)

    # Train the model
    trainer.train()


if __name__ == "__main__":
    main()
