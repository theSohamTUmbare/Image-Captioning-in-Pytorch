import torch
from torch import nn, optim
from dataset import create_dataloader  
from dataset.image_feature_extractor import extract_features  
from dataset.caption import load_captions
from imgcap_model import ImageCaptioningModel  

# Define constants
BASE_DIR = '/kaggle/input/flickr8k'
WORKING_DIR = '/kaggle/working'    
BATCH_SIZE = 16
MAX_LENGTH = 50
TEST_SPLIT = 0.1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load captions and features
mapp = load_captions(BASE_DIR)  # A function that loads and cleans captions
features = extract_features(BASE_DIR)  # A function that extracts image features

# Create dataloaders and vocabulary
train_loader, test_loader, vocab, test_ids = create_dataloader(mapp, features, batch_size=BATCH_SIZE, max_length=MAX_LENGTH, test_split=TEST_SPLIT)


# Initialize model, loss function, optimizer
vocab_size = len(vocab.word2idx) + 1  # including padding token
model = ImageCaptioningModel(vocab_size=vocab_size, max_length=MAX_LENGTH)
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 15
print_interval = 5 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

for epoch in range(epochs):
    model.train()  
    total_loss = 0

    for batch_idx, (features, sequences) in enumerate(train_loader):
        features, sequences = features.to(device), sequences.to(device)
        
        B, N, L = sequences.shape  # B = batch size, N = No. of captions, L = Length of the caption

        optimizer.zero_grad()
#         print(f"seqsize {sequences.size()}")  # Debug the shape of sequences: (batch_size, NO_CAP, seq_length)

        # Initialize tensor to collect all predictions for each timestep
        all_outputs = torch.zeros(B, N, L - 1, vocab_size).to(device)  # (batch_size, NO_CAP, seq_len-1, vocab_size)

        # Forward pass through the model for all time steps (word by word)
        for t in range(L - 1):  # Exclude the last token for training
            # Take the previous words from the sequence (up to the current step)
            input_seq = sequences[:, :, :t + 1]  # Shape (batch_size, NO_CAP, t + 1)

            # Forward pass through the model to predict the next token
            outputs = model(features, input_seq)  # Shape (batch_size, vocab_size)

            # Collect the logits for the next word prediction for all captions at timestep t
            all_outputs[:, :, t, :] = outputs  # Store the logits (batch_size, NO_CAP, t, vocab_size)

        # Flatten the target sequence (excluding the start token)
        target = sequences[:, :, 1:].reshape(-1)  # Target is the next word for all captions (flattened)

        # Reshape outputs to match the target (batch_size * NO_CAP * (seq_len - 1), vocab_size)
        all_outputs = all_outputs.view(-1, vocab_size)

        # Calculate the loss using the criterion
        loss = criterion(all_outputs, target)

        # Check if loss requires gradients (it should)
#         print(f"Loss requires_grad: {loss.requires_grad}")

        # Backpropagation
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Debugging: Print some outputs and target for the first batch of every 5th epoch
        if epoch % print_interval == 0 and batch_idx == 0:
            print(f"Epoch {epoch + 1}, Batch {batch_idx + 1}")

            # Get the predicted word indexes for the first sequence in the batch
            predicted_idx = torch.argmax(all_outputs.view(B, N, L - 1, -1), dim=-1)

            # Convert the predicted and target sequences for debugging purposes
            predicted_caption = vocab.seq_to_caption(predicted_idx[0][0].cpu().numpy())  # First caption of first image
            target_caption = vocab.seq_to_caption(target.view(B, N, L - 1)[0][0].cpu().numpy())

            print(f"Predicted Caption: {predicted_caption}")
            print(f"Target Caption: {target_caption}")

    print(f"Epoch {epoch + 1} | Loss: {total_loss / len(train_loader)}")

