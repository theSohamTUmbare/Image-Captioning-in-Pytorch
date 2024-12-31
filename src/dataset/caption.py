import os
import re
from tqdm import tqdm
from dataset.image_feature_extractor import BASE_DIR


def load_and_process_captions(captions_file_path):
    """
    Load captions from the file and map them to image IDs.

    Args:
        captions_file_path (str): Path to the captions file.

    Returns:
        dict: A dictionary mapping image IDs to cleaned captions.
    """
    # Loading captions from the file
    with open(captions_file_path, 'r') as f:
        next(f)  # Skip the header line
        captions_doc = f.read()

    # Mapping image IDs to captions
    mapp = {}
    for line in tqdm(captions_doc.split('\n'), desc="Processing captions"):
        # Splitting by comma
        tokens = line.split(',')
        if len(tokens) < 2:  
            continue
        img_id, caption = tokens[0], " ".join(tokens[1:])  # Combineing list elements into a string
        img_id = img_id.split('.')[0]  

        # Storeing the caption
        if img_id not in mapp:
            mapp[img_id] = []
        mapp[img_id].append(caption.strip())  # Storeing caption after stripping whitespace

    # Cleaning captions
    clean_captions(mapp)

    return mapp

def clean_captions(mapp):
    """
    Clean captions by applying preprocessing steps.

    Args:
        mapp (dict): A dictionary mapping image IDs to their captions.

    Returns:
        None
    """
    for key, captions in mapp.items():
        for i in range(len(captions)):
            # one caption at a time
            caption = captions[i]

            caption = caption.lower()  
            caption = re.sub(r'[^a-zA-Z\s]', '', caption)  
            caption = re.sub(r'\s+', ' ', caption)  
            caption = 'startseq ' + " ".join([word for word in caption.split() if len(word) > 1]) + ' endseq'

            captions[i] = caption

if __name__ == "__main__":
    captions_path = os.path.join(BASE_DIR, 'captions.txt')
    captions_mapping = load_and_process_captions(captions_path)
