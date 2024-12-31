import torch
import os
from PIL import Image
import matplotlib.pyplot as plt
from predict import predict_caption
from train import BASE_DIR, mapp, model, features, vocab, MAX_LENGTH
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def generate_caption(image_name):

    image_id = image_name.split('.')[0]  # the image ID from the file name
    img_path = os.path.join(BASE_DIR, "Images", image_name) 
    image = Image.open(img_path) 
    captions = mapp[image_id]  # actual captions for the image ID
    
    print("===================== ACTUAL =====================")  
    for caption in captions:
        print(caption)  

    # Predicting the caption 
    y_pred = predict_caption(model, features[image_id], vocab, MAX_LENGTH, device) 
    print("-------------------- PREDICTED ---------------------")  
    print(y_pred) 
    
    #  image
    plt.imshow(image)
    plt.axis('off')  
    plt.show()  


def main():
    image_name = '1002674143_1b742ab4b8.jpg'
    generate_caption(image_name)

if __name__ == "__main__":
    main()

