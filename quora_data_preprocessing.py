import os
import nltk
import pickle
import editdistance
from transformers import BertTokenizer
from utils import load_pickle

def create_vocab(data_paths, valid=True, min_count=3):
    """
    Create a vocabulary from the training and validation data.
    """
    word2count = {}

    def process_file(file_path):
        with open(file_path, encoding='utf-8') as f:
            for line in f:
                words = nltk.word_tokenize(line.lower())
                tags = list(zip(*nltk.pos_tag(words)))[1]
                for token in words + tags:
                    word2count[token] = word2count.get(token, 0) + 1

    process_file(data_paths['train_src'])
    process_file(data_paths['train_trg'])

    if valid:
        process_file(data_paths['valid_src'])
        process_file(data_paths['valid_trg'])

    word2idx = {'PAD': 0, 'UNK': 1, 'SOS': 2, 'EOS': 3}
    idx = 4
    for word, count in word2count.items():
        if count >= min_count:
            word2idx[word] = idx
            idx += 1

    # Save the vocabulary
    os.makedirs(data_paths['output_dir'], exist_ok=True)
    with open(os.path.join(data_paths['output_dir'], "word2idx.pkl"), 'wb') as f:
        pickle.dump(word2idx, f)

    print(f"Vocabulary created with {len(word2idx)} entries.")


def token_to_idx(data_paths, subset):
    """
    Convert tokens in source and target files to their respective indices.
    """
    word2idx = load_pickle(data_paths['word2idx'])
    src_path = data_paths[f"{subset}_src"]
    trg_path = data_paths[f"{subset}_trg"]
    output_dir = os.path.join(data_paths['output_dir'], subset)

    os.makedirs(output_dir, exist_ok=True)
    src_indices, trg_indices = {'tags': [], 'words': []}, {'tags': [], 'words': []}

    def process_file(file_path, index_store):
        with open(file_path, encoding='utf-8') as f:
            for line in f:
                words = nltk.word_tokenize(line.lower())
                tags = list(zip(*nltk.pos_tag(words)))[1]
                index_store['words'].append([word2idx.get(w, word2idx['UNK']) for w in words])
                index_store['tags'].append([word2idx.get(t, word2idx['UNK']) for t in tags])

    process_file(src_path, src_indices)
    process_file(trg_path, trg_indices)

    with open(os.path.join(output_dir, "src.pkl"), 'wb') as f:
        pickle.dump(src_indices, f)
    with open(os.path.join(output_dir, "trg.pkl"), 'wb') as f:
        pickle.dump(trg_indices, f)

    print(f"Token indices saved for {subset} subset.")


def id_to_bertid(data_paths, subset):
    """
    Convert token indices to BERT token IDs.
    """
    idx2word = load_pickle(data_paths['idx2word'])
    src_path = os.path.join(data_paths['output_dir'], subset, "src.pkl")
    trg_path = os.path.join(data_paths['output_dir'], subset, "trg.pkl")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def convert_to_text(file_path):
        data = load_pickle(file_path)
        return [' '.join([idx2word[word] for word in sentence]) for sentence in data['words']]

    src_text = convert_to_text(src_path)
    trg_text = convert_to_text(trg_path)

    bert_src = [tokenizer.encode(text, add_special_tokens=True) for text in src_text]
    bert_trg = [tokenizer.encode(text, add_special_tokens=True) for text in trg_text]

    output_dir = os.path.join(data_paths['output_dir'], subset)
    with open(os.path.join(output_dir, "bert_src.pkl"), 'wb') as f:
        pickle.dump(bert_src, f)
    with open(os.path.join(output_dir, "bert_trg.pkl"), 'wb') as f:
        pickle.dump(bert_trg, f)

    print(f"BERT tokenized data saved for {subset} subset.")


def find_examples(data_paths, subset):
    """
    Find similar examples based on tag similarity.
    """
    tags = load_pickle(os.path.join(data_paths['output_dir'], subset, "src.pkl"))
    trgs = load_pickle(os.path.join(data_paths['output_dir'], subset, "trg.pkl"))
    similar_list = []

    for i, target in enumerate(trgs['words']):
        if i % 1000 == 0:
            print(f"Processing {i}/{len(trgs['words'])}...")
        similarities = []
        for j, compare in enumerate(trgs['words']):
            if i == j or abs(len(target) - len(compare)) > 1:
                continue
            dist = editdistance.eval(target, compare)
            similarities.append((j, dist))
        best_matches = sorted(similarities, key=lambda x: x[1])[:5]
        similar_list.append([idx for idx, _ in best_matches])

    with open(os.path.join(data_paths['output_dir'], subset, "sim.pkl"), 'wb') as f:
        pickle.dump(similar_list, f)

    print(f"Similar examples saved for {subset} subset.")


def main():
    # Define paths
    data_paths = {
        "train_src": "dataset-text/quora-text/quora/train_src.txt",
        "train_trg": "dataset-text/quora-text/quora/train_trg.txt",
        "valid_src": "dataset-text/quora-text/quora/valid_src.txt",
        "valid_trg": "dataset-text/quora-text/quora/valid_trg.txt",
        "test_src": "dataset-text/quora-text/quora/test_src.txt",
        "test_trg": "dataset-text/quora-text/quora/test_trg.txt",
        "word2idx": "datasets/processed/qqp-pos/data/word2idx.pkl",
        "idx2word": "datasets/processed/qqp-pos/data/idx2word.pkl",
        "output_dir": "datasets/processed/qqp-pos/data/"         
    }

    # Execute functions
    create_vocab(data_paths)
    for subset in ['train', 'valid', 'test']:
        token_to_idx(data_paths, subset)
        id_to_bertid(data_paths, subset)
        find_examples(data_paths, subset)


if __name__ == "__main__":
    main()
