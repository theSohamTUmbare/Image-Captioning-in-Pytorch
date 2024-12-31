import tqdm
from dataset.dataset import create_dataloader
from nltk.translate.bleu_score import corpus_bleu
from predict import predict_caption
from train import test_ids, model, features, mapp, vocab, MAX_LENGTH, device

# Validate with test data 
def test(model, features, mapp, vocab, max_length, device):
    actual, predicted = list(), list()
    for key in tqdm(test_ids):
        # Get actual captions
        captions = mapp[key]
        
        # Predict the caption for the image
        y_pred = predict_caption(model, features[key], vocab, max_length, device)
        
        # Prepare the actual and predicted captions for BLEU score calculation
        actual_captions = [caption.split() for caption in captions]
        y_pred = y_pred.split()
        
        actual.append(actual_captions)
        predicted.append(y_pred)
    
    return predicted, actual


def main():
    predicted, actual = test(model, features, mapp, vocab, MAX_LENGTH, device)
    
    # The BLEU scores
    print("BLEU-1: %f" % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
    print("BLEU-2: %f" % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))


if __name__ == "__main__":
    main()
