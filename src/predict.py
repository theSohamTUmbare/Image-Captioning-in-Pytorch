import json
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from dataset.dataset import Vocabulary
from imgcap_model import ImageCaptioningModel 

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import io

app = FastAPI()

def predict_caption(model, img_path, feature_extraction_model, vocab, device, max_length=34):
   
    feature_extraction_model.eval()  
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize for ImageNet
    ])
    
    image = Image.open(img_path).convert('RGB')  # Converting to RGB to avoid issues with single-channel images            
    image = transform(image)
    image = image.unsqueeze(0)  
    # Extracting features using the feature_extraction_model
    with torch.no_grad():
        img_feature = feature_extraction_model(image)              


    model.eval()
    
    # Converting image_features to a PyTorch tensor and move it to the appropriate device
    image_features = torch.tensor(img_feature, dtype=torch.float32).to(device)
    image_features = image_features.squeeze(-1)
    image_features = image_features.squeeze(-1)

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


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    # Read and validate the uploaded file
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert('RGB')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")
    feature_extraction_model = models.resnet50(pretrained=True)
    feature_extraction_model = nn.Sequential(*list(feature_extraction_model.children())[:-1])     # Modified the model to output the second to last layer

    
    with open("word2idx.json", "r") as f_w2i:
        w2i = json.load(f_w2i)

    with open("idx2word.json", "r") as f_i2w:
        i2w_strkeys = json.load(f_i2w)

    vocab = Vocabulary()
    vocab.load_from_dicts(w2i, i2w_strkeys)
        
    model = ImageCaptioningModel(max_length=34, vocab_size=8768)
    
    checkpoint_path = 'model_epoch_100.pth'
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    try:
        caption = predict_caption(
            model=model,
            img_path=io.BytesIO(contents),               # pass a file-like
            feature_extraction_model=feature_extraction_model,
            vocab=vocab,
            device=device,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Captioning failed: {e}")

    return JSONResponse(content={"caption": caption})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "predict:app",    # file name predict.py
        host="0.0.0.0",
        port=8000,
        reload=True       # auto-reload 
    )


# def main():
    
#     feature_extraction_model = models.resnet50(pretrained=True)
#     feature_extraction_model = nn.Sequential(*list(feature_extraction_model.children())[:-1])     # Modified the model to output the second to last layer

    
#     with open("word2idx.json", "r") as f_w2i:
#         w2i = json.load(f_w2i)

#     with open("idx2word.json", "r") as f_i2w:
#         i2w_strkeys = json.load(f_i2w)

#     vocab = Vocabulary()
#     vocab.load_from_dicts(w2i, i2w_strkeys)
        
#     model = ImageCaptioningModel(max_length=34, vocab_size=8768)
    
#     checkpoint_path = 'model_epoch_100.pth'
#     checkpoint = torch.load(checkpoint_path, map_location='cpu')
#     model.load_state_dict(checkpoint['model_state_dict'])
    
#     image_name = '1002674143_1b742ab4b8.jpg'
#     cap = predict_caption(model, img_path="data/runningDog.png", feature_extraction_model=feature_extraction_model, vocab=vocab, device='cpu')
    
#     print("---Caption----------------------------------------------------------------------------------------------")
#     print(cap)
#     print("--------------------------------------------------------------------------------------------------------")
    
# if __name__ == "__main__":
#     main()
