import torch
from tokenizer import Tokenizer
from model import BharatAI
import pickle

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load tokenizer and model
tokenizer = Tokenizer('tokenizer.model')
with open('model.pkl', 'rb') as f:
    model = pickle.load(f).to(device)

def generate_text(prompt, max_length=100):
    encoded_prompt = torch.tensor([tokenizer.encode(prompt, bos=True)], dtype=torch.long).to(device)
    output = model.generate(encoded_prompt, max_new_tokens=max_length)
    return tokenizer.decode(output[0].tolist())