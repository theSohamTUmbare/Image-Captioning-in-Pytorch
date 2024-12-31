import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm

def extract_image_features(base_dir, valid_extensions=None, device=None):
    """
    Extract features from images in a specified directory using a pretrained ResNet50 model.

    Parameters:
        base_dir (str): Path to the directory containing images.
        valid_extensions (list, optional): List of valid image file extensions. Defaults to ['.jpg', '.jpeg', '.png'].
        device (torch.device, optional): Device to run the model on (e.g., 'cuda' or 'cpu'). Defaults to auto-detected device.

    Returns:
        dict: A dictionary where keys are image IDs and values are feature tensors.
    """
    if valid_extensions is None:
        valid_extensions = ['.jpg', '.jpeg', '.png']

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Loading ResNet50 model and modify it to output second-to-last layer features
    model = models.resnet50(pretrained=True)
    model = nn.Sequential(*list(model.children())[:-1])  # Excluding the last fully connected layer
    model = model.to(device)
    model.eval()  # Set the model to evaluation mode

    # transformations for preprocessing images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize for ImageNet
    ])

    # Dictionary to store extracted features
    features = {}

    for img_name in tqdm(os.listdir(base_dir), desc="Extracting features"):
        img_path = os.path.join(base_dir, img_name)

        # Check for valid image extension
        if any(img_name.lower().endswith(ext) for ext in valid_extensions):
            try:
                image = Image.open(img_path).convert('RGB')
                image = transform(image).unsqueeze(0).to(device)  # Applying transformations and batch dimension

                # Extracting features using the model
                with torch.no_grad():
                    img_feature = model(image)

                # Extracting image ID (filename without extension)
                image_id = os.path.splitext(img_name)[0]

                # Storeing the features in the dictionary
                features[image_id] = img_feature.cpu().numpy()

            except Exception as e:
                print(f"An error occurred with image {img_name}: {e}")
        else:
            print(f"Skipping non-image file: {img_name}")

    return features