import torch
import torch.nn as nn
import torch.nn.functional as F

class ImageCaptioningModel(nn.Module):
    def __init__(self, vocab_size, max_length, feature_size=2048, embed_size=256, hidden_size=256):
        super(ImageCaptioningModel, self).__init__()
        self.image_fc = nn.Linear(feature_size, embed_size)
        self.caption_embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)  # Project hidden state to vocab size
        self.max_length = max_length  # Store max_length
        

    def forward(self, image_features, captions):
#         print(f"Caption shape: {captions.size()}")
        B, N, T = captions.shape  # B = batch size, N = number of captions, T = length of each caption
        
        # Embed the image features for all captions
        img_embedded = self.image_fc(image_features)  # (batch_size, embed_size)
        
        # Repeat the image embedding for each caption
        img_embedded = img_embedded.unsqueeze(1)  # (batch_size, 1, embed_size)
        img_embedded = img_embedded.repeat(1, N, 1)  # (batch_size, N, embed_size)
        
        # Reshape to match the number of captions
        img_embedded = img_embedded.view(B * N, 1, -1)  # (batch_size * N, 1, embed_size)
        img_embedded = img_embedded.repeat(1, T, 1)  # (batch_size * N, T, embed_size)
        
        # Embed the captions
        cap_embedded = self.caption_embedding(captions)  # (batch_size, N, T, embed_size)
        
        # Flatten the batch and caption dimensions to process them together
        cap_embedded = cap_embedded.view(B * N, T, -1)  # (batch_size * N, T, embed_size)
        
        
        # Combine image and caption embeddings
#         combined_input = torch.cat((img_embedded, cap_embedded), dim=1)  # (batch_size * N, T + 1, embed_size)
        combined_input = img_embedded + cap_embedded  # (batch_size*N, T, embed_size)


        # Pass through the LSTM
        lstm_out, _ = self.lstm(combined_input)  # (batch_size * N, T + 1, hidden_size)
        
        # Take the last hidden state (for next-token prediction)
        last_lstm_out = lstm_out[:, -1, :]  # (batch_size * N, hidden_size)
        
        # Pass the last hidden state through the fully connected layer to get vocab size logits
        output = self.fc(last_lstm_out)  # (batch_size * N, vocab_size)

        return output.view(B, N, -1)  # Reshape back to (batch_size, N, vocab_size)
 