import torch

import argparse

from name_classifier.data.utils import line_to_tensor
from name_classifier.utils import load_model
from name_classifier.models.rnn import RNN
import name_classifier.metadata.names as metadata

def _setup_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--input",
        type = str,
        default = "alpha",
        help = "The name to classify."
    )
    
    return parser

def predict(model, input, n_predictions = 3):
    model.eval()
    with torch.inference_mode():
        input_tensor = line_to_tensor(input)
        
        hidden = model.init_hidden()
        
        for i in range(input_tensor.size()[0]):
            output, hidden = model(input_tensor[i], hidden)
            
    top_v, top_i = output.topk(n_predictions, 1, True)
    
    predictions = {}
    
    for i in range(n_predictions):
        value = top_v[0][i].item()
        idx = top_i[0][i].item()
        predictions[metadata.CATEGORIES[idx]] = value
    
    return predictions
        
def main():
    parser = _setup_parser()
    args = parser.parse_args()
    
    model = RNN(metadata.N_LETTERS, 128, metadata.N_CATEGORIES)
    load_model(model, "models/simple_rnn.pth")
    print(predict(model, args.input))
    

if __name__ == "__main__":
    main()