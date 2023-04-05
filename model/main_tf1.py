from data_loading import DataLoader
from tf1_model import CAPEmodel, Trainer, RUNTIME_SETTINGS, HYPERPARAMS

def main():    
    # Load the data
    loaded_data = DataLoader(RUNTIME_SETTINGS, HYPERPARAMS)

    # Initialize the model
    model = CAPEmodel(loaded_data, HYPERPARAMS)

    # Initialize the trainer
    trainer = Trainer(model, RUNTIME_SETTINGS)

    # Train the model
    trainer.train()


if __name__ == "__main__":
    main()
