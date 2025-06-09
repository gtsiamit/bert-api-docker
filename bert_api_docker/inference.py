from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    PreTrainedModel,
)
from sklearn.preprocessing import LabelEncoder
from torch import Tensor
from pathlib import Path
import pickle
import torch


file_path = Path(__file__).parent
MODEL_PATH = str(file_path / "fine_tuned")
LE_PATH = str(file_path / "fine_tuned/le.pkl")


class TextClassification:
    """
    A class for text classification using a pre-trained BERT model.
    """

    def __init__(self, model_path: str = MODEL_PATH, le_path: str = LE_PATH):
        """
        Initializes the TextClassification class with the specified model and label encoder paths.

        Args:
            model_path (str, optional): Path to the pre-trained model. Defaults to MODEL_PATH.
            le_path (str, optional): Path to the label encoder file. Defaults to LE_PATH.
        """

        # Set the model and label encoder paths
        self.model_path = model_path
        self.le_path = le_path

        # Load the tokenizer, model, and label encoder
        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()

        self.le = self._load_le()

    def _load_tokenizer(self) -> PreTrainedTokenizerBase:
        """
        Loads the pretrained tokenizer from the specified model path.

        Returns:
            PreTrainedTokenizerBase: The loaded tokenizer.
        """

        return AutoTokenizer.from_pretrained(self.model_path)

    def _load_model(self) -> PreTrainedModel:
        """
        Loads the pretrained sequence classification model from the specified path.

        Returns:
            PreTrainedModel: The loaded BERT-based model set to evaluation mode.
        """

        model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        _ = model.eval()

        return model

    def _load_le(self) -> LabelEncoder:
        """
        Loads the fitted label encoder from the specified pickle file path.

        Returns:
            LabelEncoder: The loaded label encoder used to decode predictions.
        """

        with open(self.le_path, "rb") as f:
            le = pickle.load(f)

        return le

    def _get_probs(self, text: str) -> Tensor:
        """
        Tokenizes input text and computes prediction probabilities.

        Args:
            text (str): The input text to classify.

        Returns:
            Tensor: A tensor of softmax probabilities for each class.
        """

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=256,
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1)

        return probs

    def _probs_to_class(self, probs: Tensor) -> str:
        """
        Converts softmax probabilities into a predicted class label.

        Args:
            probs (Tensor): A tensor of class probabilities.

        Returns:
            str: The decoded class label.
        """

        pred_class = torch.argmax(probs, dim=1).item()

        pred_class = self.le.inverse_transform([pred_class])[0]

        return pred_class

    def predict(self, _input: str) -> str:
        """
        Predicts the class label for a given input text.

        Args:
            _input (str): The input text.

        Returns:
            str: The predicted class label.
        """

        probs = self._get_probs(text=_input)

        pred_class = self._probs_to_class(probs=probs)

        return pred_class
