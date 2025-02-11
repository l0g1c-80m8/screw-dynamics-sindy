from model import LSTMModel, MLP
import torch
from loguru import logger
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.dataloader import ScrewdrivingDataset


class ModelTrainer:
    def __init__(
        self,
        model: LSTMModel | MLP,
        criterion: torch.nn.MSELoss | torch.nn.L1Loss,
        optimizer: torch.optim.Optimizer,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

        self.epochs = 1000
        self.batch_size = 20
        logger.info(
            f"Initialized Model Trainer with Following Params: Epochs: {self.epochs}, Batch Size: {self.batch_size}"
        )

    def train(self, train_loader, val_loader, device):
        for epoch in tqdm(range(self.epochs), "Epoch: "):
            total_loss = 0
            self.model.train()

            for data in tqdm(train_loader, "Training Batch: "):
                inputs, targets = data
                inputs, targets = inputs.to(device), targets.to(device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            logger.info(
                f"Epoch {epoch + 1}/{self.epochs}, Training Loss: {total_loss / len(train_loader)}"
            )
            self.evaluate(val_loader=val_loader, device=device)

        return total_loss / len(train_loader)

    def evaluate(self, val_loader, device):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for data in val_loader:
                inputs, targets = data
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                total_loss += loss.item()
        logger.info(f"Validation Loss: {total_loss / len(val_loader)}")
        return total_loss / len(val_loader)


def evaluate_test_loss(model, test_loader, device):
    """Evaluates test MSE loss."""
    model.eval()
    total_loss = 0
    criterion = torch.nn.MSELoss()  # MSE loss

    with torch.no_grad():
        for data in test_loader:
            inputs, targets = data
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)  # Compute MSE

            total_loss += loss.item()

    test_loss = total_loss / len(test_loader) if len(test_loader) > 0 else float("inf")
    logger.info(f"Test MSE Loss: {test_loss}")
    return test_loss


def run_training(model, model_name, train_loader, val_loader, device, learning_rate, checkpoint_path):
    """Trains and saves the model."""
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()
    trainer = ModelTrainer(model, criterion, optimizer)

    logger.info(f"Training {model_name} Model:\n{model}")
    trainer.train(train_loader, val_loader, device)

    # Save trained model
    torch.save(model.state_dict(), checkpoint_path)
    logger.info(f"Saved {model_name} model to {checkpoint_path}")

    final_loss = trainer.evaluate(val_loader, device)
    return final_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Black Box NN/GRU based model")
    parser.add_argument("--model", type=str, default="mlp", choices=["lstm", "mlp"], help="Model type: lstm or mlp")
    parser.add_argument("--data_dir", type=str, default="./out/sindy-data", help="Data directory")
    parser.add_argument("--out_dir", type=str, default="./out/sindy-model-out", help="Output directory")
    parser.add_argument("--train_file", type=str, default="train_data.csv", help="Train file name")
    parser.add_argument("--val_file", type=str, default="val_data.csv", help="Validation file name")
    parser.add_argument("--test_file", type=str, default="test_data.csv", help="Test file name")

    parser.add_argument("--input_var_dim", type=int, default=17, help="Input variable dimension")
    parser.add_argument("--state_var_dim", type=int, default=2, help="State variable dimension")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run operations on")
    parser.add_argument("--learning_rate", type=float, default=0.00001, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of epochs")
    parser.add_argument('--window_length', type=int, default=1, help='batch window size')

    args = parser.parse_args()
    logger.info(f"Arguments: {args}")

    train_loader = DataLoader(ScrewdrivingDataset(mode="train", **vars(args)))
    val_loader = DataLoader(ScrewdrivingDataset(mode="test", **vars(args)))
    test_loader = DataLoader(ScrewdrivingDataset(mode="test", **vars(args)))  # Separate test set

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # --- Train LSTM Model ---
    lstm_model = LSTMModel(input_dim=17, hidden_dim=34, output_dim=2, num_layers=3)
    lstm_model.to(device)
    lstm_loss = run_training(lstm_model, "LSTM", train_loader, val_loader, device, args.learning_rate, "./out/baseline/lstm_model.pth")

    # --- Train MLP Model ---
    mlp_model = MLP(input_dim=17, hidden_dim=34, output_dim=2)
    mlp_model.to(device)
    mlp_loss = run_training(mlp_model, "MLP", train_loader, val_loader, device, args.learning_rate, "./out/baseline/mlp_model.pth")

    # --- Load and Evaluate on Test Data ---
    lstm_model.load_state_dict(torch.load("./out/baseline/lstm_model.pth"))
    mlp_model.load_state_dict(torch.load("./out/baseline/mlp_model.pth"))

    # --- Print Final Outputs ---
    print("\nFinal Evaluation Results:")
    print(f"LSTM Model Final Validation Loss: {lstm_loss}")
    print(f"MLP Model Final Validation Loss: {mlp_loss}")

    lstm_test_loss = evaluate_test_loss(lstm_model, test_loader, device)
    mlp_test_loss = evaluate_test_loss(mlp_model, test_loader, device)

    print("\nFinal Test Results:")
    print(f"LSTM Model Test MSE Loss: {lstm_test_loss}")
    print(f"MLP Model Test MSE Loss: {mlp_test_loss}")
