import torch
import pickle
import torch.optim as optim
from model import BharatAI
from tokenizer import Tokenizer

# Device configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load tokenizer
tokenizer = Tokenizer('botchan.model')

# Define batch retrieval function
def get_batch(split):
    # Placeholder function - replace with actual data loading logic
    batch_size, block_size = 16, 256  # Example batch size and block size
    data = torch.randint(0, 1000, (batch_size, block_size), dtype=torch.long).to(device)
    target = torch.randint(0, 1000, (batch_size, block_size), dtype=torch.long).to(device)
    return data, target

# Train model function
def train_model(model, optimizer, epochs=250):
    model.train()
    for epoch in range(epochs):
        xb, yb = get_batch('train')
        optimizer.zero_grad()
        logits, loss = model(xb, yb)
        loss.backward()
        optimizer.step()
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch}: Loss {loss.item()}")
    
    # Save trained model
    with open('model-latest.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("Model saved!")

# Initialize and train model
if __name__ == '__main__':
    vocab_size = 1000  # Adjust as per your tokenizer
    model = BharatAI(vocab_size).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)
    train_model(model, optimizer)