from Data import DummyTextDataset
from torch import optim, nn
import torch
from torch.utils.data import DataLoader

from Models.build_model import build_model

LEARNING_RATE = 1e-5
WEIGHT_DECAY = 5e-4
VOCAB_SIZE = 100
SEQ_LEN = 50
NUM_SAMPLES = 500
NUM_EPOCHS = 5
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET = DummyTextDataset(vocab_size=VOCAB_SIZE, seq_len=SEQ_LEN, num_samples=NUM_SAMPLES)
DATALOADER = DataLoader(DATASET, batch_size=BATCH_SIZE, shuffle=True)


def train(model, optimizer, criterion):
    model.train()
    epoch_loss = 0.0

    for src, target in DATALOADER:
        src = src.to(DEVICE)
        target = target.to(DEVICE)

        # Teacher forching: shift target tokens for decoder input
        target_input = target[:, :-1]
        target_output = target[:, 1:]

        # Forward pass
        output, _ = model(src, target_input)

        # Compute loss
        y_hat = output.contiguous().view(-1, VOCAB_SIZE)
        y_gt = target_output.contiguous().view(-1)
        loss = criterion(y_hat, y_gt)
        epoch_loss += loss.item()

        # Backpropagatio
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return epoch_loss / NUM_SAMPLES

def main():
    model = build_model(VOCAB_SIZE, VOCAB_SIZE)
    optimizer = optim.Adam(params=model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(NUM_EPOCHS):
        train_loss = train(model, optimizer, criterion)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {train_loss / len(DATALOADER):.4f}")

if __name__=="__main__":
    torch.manual_seed(0)
    main()