import gradio as gr

from name_classifier.models.rnn import RNN
import name_classifier.metadata.names as metadata
from name_classifier import utils
from name_classifier.predict import predict

model = RNN(metadata.N_LETTERS, 128, metadata.N_CATEGORIES)
utils.load_model(model, "models/simple_rnn.pth")

def predict_language(name):
    predictions = predict(model, name, n_predictions = 3)
    total = sum(predictions.values())

    for key, value in predictions.items():
        predictions[key] = 1 - value / total
    
    return predictions


demo = gr.Interface(
    fn = predict_language,
    inputs = ["text"],
    outputs = [gr.Label()],
)

demo.launch()