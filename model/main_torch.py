from data_loading import DataLoader
from pytorch_model import CAPEmodel, Trainer
from config import RUNTIME_SETTINGS, HYPERPARAMS

def main():    
    # Load the data
    loaded_data = DataLoader(RUNTIME_SETTINGS, HYPERPARAMS)

    # Initialize the model
    model = CAPEmodel(HYPERPARAMS)

    # Initialize the trainer
    trainer = Trainer(model, RUNTIME_SETTINGS)

    # Train the model
    trainer.train(loaded_data)


if __name__ == "__main__":
    main()
