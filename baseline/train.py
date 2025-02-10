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


def run_training(model, model_name, train_loader, val_loader, device, learning_rate):
    # Create optimizer and criterion for this model
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()
    trainer = ModelTrainer(model, criterion, optimizer)
    logger.info(f"Training {model_name} Model:\n{model}")
    trainer.train(train_loader, val_loader, device)
    final_loss = trainer.evaluate(val_loader, device)
    return final_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Black Box NN/GRU based model")
    parser.add_argument(
        "--model",
        type=str,
        default="mlp",
        choices=["lstm", "mlp"],
        help="Model type: lstm or mlp",
    )
    parser.add_argument('--data_dir', type=str, default='./out/sindy-data',
                        action='store', dest='data_dir', help='data directory')
    parser.add_argument('--out_dir', type=str, default='./out/sindy-model-out',
                        action='store', dest='out_dir', help='output directory')
    parser.add_argument('--train_file', type=str, default='train_data.csv',
                        action='store', dest='train_file', help='train file name')
    parser.add_argument('--val_file', type=str, default='val_data.csv',
                        action='store', dest='val_file', help='val file name')
    parser.add_argument('--test_file', type=str, default='test_data.csv',
                        action='store', dest='test_file', help='test file name')

    parser.add_argument('--poly_order', type=int, default=3,
                        action='store', dest='poly_order', help='highest polynomial order in sindy library')
    parser.add_argument('--include_constant', type=bool, default=True,
                        action='store', dest='include_constant', help='include constant function in sindy library')
    parser.add_argument('--use_sine', type=bool, default=True,
                        action='store', dest='use_sine', help='use sine function in sindy library')

    parser.add_argument('--input_var_dim', type=int, default=17,
                        action='store', dest='input_var_dim', help='dimension of input variable')
    parser.add_argument('--state_var_dim', type=int, default=2,
                        action='store', dest='state_var_dim', help='dimension of state variable')

    parser.add_argument('--device', type=str, default='cpu',
                        action='store', dest='device', help='device to run operations on')

    parser.add_argument('--learning_rate', type=float, default=.00001,
                        action='store', dest='learning_rate', help='learning rate for training')
    parser.add_argument('--weight_decay', type=float, default=.000001,
                        action='store', dest='weight_decay', help='weight decay for training')
    # parser.add_argument('--epochs', type=int, default=50,
    #                     action='store', dest='epochs', help='epochs for training')

    parser.add_argument('--window_length', type=int, default=1,
                        action='store', dest='window_length', help='batch window size')

    parser.add_argument("--epochs", type=int, default=1000, help="Input dimension")
    args = parser.parse_args()
    logger.info(f"Arguments: {args}")
    logger.info(f"Arguments: {parser.parse_args()}")

    train_loader = DataLoader(ScrewdrivingDataset(mode='train', **vars(args)))
    val_loader = DataLoader(ScrewdrivingDataset(mode='test', **vars(args)))
    assert train_loader is not None, "Train loader is None"
    assert val_loader is not None, "Validation loader is None"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # --- Train LSTM Model ---
    lstm_model = LSTMModel(input_dim=17, hidden_dim=34, output_dim=2, num_layers=3)
    lstm_model.to(device)
    lstm_loss = run_training(lstm_model, "LSTM", train_loader, val_loader, device, args.learning_rate)

    # --- Train MLP Model ---
    # IMPORTANT: Ensure your MLP definition is corrected. For example, the second linear layer should accept hidden_dim as input.
    mlp_model = MLP(input_dim=17, hidden_dim=34, output_dim=2)
    mlp_model.to(device)
    mlp_loss = run_training(mlp_model, "MLP", train_loader, val_loader, device, args.learning_rate)

    # --- Print Final Outputs ---
    print("\nFinal Evaluation Results:")
    print(f"LSTM Model Final Validation Loss: {lstm_loss}")
    print(f"MLP Model Final Validation Loss: {mlp_loss}")
