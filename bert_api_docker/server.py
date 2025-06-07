from fastapi import FastAPI
from pydantic import BaseModel, Field
from inference import TextClassification


class InputData(BaseModel):
    # Input data model for text classification.
    text: str = Field(
        ...,
        title="Input text",
        description="The text str to classify",
        example="This is a text input.",
    )


class OutputResponse(BaseModel):
    # Output response model for text classification prediction.
    message: str = Field(
        ...,
        title="Response message",
        description="Status message for the prediction",
        example="Prediction done",
    )
    input: str = Field(
        ...,
        title="Input text",
        description="The original input text sent for prediction",
        example="This is a text input.",
    )
    predicted_class: str = Field(
        ...,
        title="Predicted class label",
        description="The class label predicted by the model",
        example="tech",
    )


# Create FastAPI app instance
app = FastAPI()

# Initialize the TextClassification model
bert_classifier = TextClassification()


# Define the prediction endpoint
@app.post("/predict", response_model=OutputResponse)
def predict(input_data: InputData) -> OutputResponse:
    """
    Predict the class of the input text using the BERT classifier.

    Args:
        input_data (InputData): Input data containing the text to classify.

    Returns:
        OutputResponse: Output response containing the prediction result.
    """

    # Get the input text from the request
    input_text = input_data.text
    # Perform prediction using the bert_classifier
    prediction = bert_classifier.predict(input_text)

    return OutputResponse(
        message="Prediction done", input=input_text, predicted_class=prediction
    )
