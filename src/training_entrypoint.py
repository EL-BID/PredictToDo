import os  # noqa: I001
import time
import logging
import argparse
from pathlib import Path

# import ModelTrainer


LOG_LEVEL = int(os.getenv('SM_LOG_LEVEL', logging.INFO))
CHANNEL_TRAIN = os.getenv('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train')
CHANNEL_VALIDATION = os.getenv('SM_CHANNEL_VALIDATION', '/opt/ml/input/data/validation')
# CHECKPOINTS_PATH = os.getenv('SM_CHECKPOINTS_PATH', '/opt/ml/checkpoints')
MODEL_DIR = os.getenv('SM_MODEL_DIR', '/opt/ml/model')

logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)


def parse_args() -> dict:
    parser = argparse.ArgumentParser()

    # Hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument(
        '--epochs',
        help='Total number of training epochs. Each epoch represents a full pass over the entire dataset.',
        type=int,
        default=10
    )
    parser.add_argument(
        '--batch',
        help='Batch size for training, indicating how many images are processed before the model\'s internal parameters are updated.',
        type=int,
        default=32
    )
    parser.add_argument(
        '--eta',
        help='Learning rate for the optimizer, controlling how much to change the model in response to the estimated error each time the model weights are updated.',
        type=float,
        default=0.1
    )
    parser.add_argument(
        '--gamma',
        help='Minimum loss reduction required to make a further partition on a leaf node of the tree. Used to control overfitting.',
        type=float,
        default=2.0
    )
    # parser.add_argument(
    #     '--project',
    #     help='Name of the project directory where training outputs are saved. Allows for organized storage of different experiments.',
    #     type=str,
    #     default=CHECKPOINTS_PATH
    # )

    payload = vars(parser.parse_args())
    logger.info(f"Parsed arguments: {payload}")
    return payload


def check_data_channels() -> tuple[Path, Path]:
    """
    Check if the data channels for training and validation are set.
    If not, raise an error.
    """
    if not (train_path := Path(CHANNEL_TRAIN)).exists():
        raise ValueError(f"Training data channel '{CHANNEL_TRAIN}' does not exist.")
    if not (validation_path := Path(CHANNEL_VALIDATION)).exists():
        raise ValueError(f"Validation data channel '{CHANNEL_VALIDATION}' does not exist.")

    logger.info(f"Training data files: {list(train_path.iterdir())}")
    logger.info(f"Validation data files: {list(validation_path.iterdir())}")

    return train_path, validation_path


def train_model(args: dict, train_path: Path = None, validation_path: Path = None) -> None:
    logger.info("Starting model training...")

    # This is where you would implement the actual training logic using the parsed arguments.
    # For example, you could use a deep learning framework like PyTorch or TensorFlow to
    # load the model, prepare the data, and run the training loop.

    # Simulate error loss and accuracy for demonstration purposes
    for epoch in range(1, 6):  # Simulating 5 epochs
        loss = 0.1 * epoch  # Example loss value
        accuracy = 0.9 - (0.1 * epoch)  # Example accuracy value
        logger.info(f"epoch={epoch}, train-loss={loss:.4f}, train-accuracy={accuracy:.4f}")
        time.sleep(2)  # Simulate time taken for each epoch
    # model = ModelTrainer.train(**args)

    logger.info("Model training completed successfully.")
    # return model


if __name__ == "__main__":
    args = parse_args()

    train_path, validation_path = check_data_channels()
    model = train_model(args, train_path, validation_path)

    model_path = Path(MODEL_DIR) / 'model.pth'
    # Here you would save your trained model, e.g., using torch.save(model.state_dict(), model_path)
    # model_path.write_text("Trained model placeholder")  # Placeholder for actual model saving
    logger.info(f"Model saved to {model_path}")
