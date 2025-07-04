import os
import torch.nn as nn
from transformers import T5EncoderModel, T5Tokenizer


class TextEncoder(nn.Module):
    def __init__(self):
        super(TextEncoder, self).__init__()
        self.tokenizer = T5Tokenizer.from_pretrained("t5-base", legacy=False)
        self.encoder = T5EncoderModel.from_pretrained("t5-base").to("cpu")

        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, x):
        inputs = self.tokenizer(x, return_tensors="pt", padding=True).to("cpu")
        outputs = self.encoder(**inputs)
        last_hidden_states = outputs.last_hidden_state
        return last_hidden_states
