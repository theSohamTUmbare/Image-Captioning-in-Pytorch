import torch

def predict_caption(model, image_features, vocab, max_length, device):
    # model to evaluation mode
    model.eval()
    
    # Converting image_features to a PyTorch tensor and move it to the appropriate device
    image_features = torch.tensor(image_features, dtype=torch.float32).to(device)
    

    # start token index is 1 (adjust if different)
    start_token = 1
    caption = [start_token]  
    
    # Generateing captions step-by-step
    for _ in range(max_length):  # Limiting to max_length tokens
        
        # Converting caption list to tensor and move to device
        input_tensor = torch.tensor(caption).unsqueeze(0).to(device)  # Shape: (1, current_length)
        
        # Embed the caption (for one caption at a time)
        input_tensor = input_tensor.unsqueeze(1)  # Shape: (1, 1, current_length)
        
        # Forward pass through the model
        output = model(image_features, input_tensor)  # Shape: (1, vocab_size)
        
        # Get the predicted token (next word) with the highest probability
        predicted_token = torch.argmax(output, dim=-1).item()  # Get the index of the predicted word
        
        # Stop if the predicted token is the end token (33 is the end token index)
        if predicted_token == 33:  
            break
        
        # Appending the predicted token to the caption
        caption.append(predicted_token)
    
    # Converting the caption indices back to words
    predicted_caption = vocab.seq_to_caption(caption)  
    
    return predicted_caption
