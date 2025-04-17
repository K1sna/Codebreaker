import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.chunk import ne_chunk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from tqdm import tqdm
import sys
sys.path.insert(0, "/home/C0debr34k3r/Codebreaker")

mode = "Starcoder2_4096"
mode = str.lower(mode)
if "starcoder2" in mode:
    from StarCoder2.starcoder2 import StarCoder2Querier
    model = StarCoder2Querier()

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')


import json

def extract_prompts(input_path):
    prompts = []
    with open(input_path, 'r', encoding='utf-8') as infile:
        for line in infile:
            data = json.loads(line.strip())
            if 'prompt' in data:
                prompts.append(data['prompt'])
    return prompts

def calculate_entropy(labels):
    label_counts = np.bincount(labels[labels >= 0])
    probabilities = label_counts / len(labels)
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

def process_response(response):
    tokens = word_tokenize(response)
    filtered_tokens = [token for token in tokens if len(token) > 3]
    tagged_tokens = pos_tag(filtered_tokens)
    entities = ne_chunk(tagged_tokens)
    extracted_tokens = [word for word, pos in tagged_tokens]
    processed_response = " ".join(extracted_tokens)
    return processed_response

def process_comments(comments):
    results = []
    for comment in comments:
        tokens = word_tokenize(comment)
        filtered_tokens = [token for token in tokens if len(token) > 3]
        tagged_tokens = pos_tag(filtered_tokens)
        entities = ne_chunk(tagged_tokens)
        extracted_tokens = [word for word, pos in tagged_tokens]
        results.append(extracted_tokens)
    return results

def calculate_semantic_entropies(responses, m=10):
    semantic_entropies = []
    for response in responses:
        texts = [response[f'response{k+1}'] for k in range(m)]
        results = process_comments(texts)
        texts = [" ".join(tokens) for tokens in results]
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(texts)
        dbscan = DBSCAN(eps=1.32, min_samples=1)
        clusters = dbscan.fit_predict(X)
        entropy = calculate_entropy(clusters)
        semantic_entropies.append(entropy)
    return semantic_entropies

def cal_sentropy(jsonl_path):
    prompts = extract_prompts(jsonl_path)
    batch_size = 12 
    total_batches = (len(prompts) + batch_size - 1) // batch_size

    semantic_entropies = []

    for batch_idx in range(total_batches):
        batch_start = batch_idx * batch_size
        batch_end = min((batch_idx + 1) * batch_size, len(prompts))
        batch_prompts = prompts[batch_start:batch_end]

        with torch.no_grad():
            responses = model.simple_batch_query(batch_prompts, 1, 10) 
            batch_entropies = calculate_semantic_entropies(responses, m=10) 
            semantic_entropies.extend(batch_entropies)

        del responses
        torch.cuda.empty_cache()

    updated_data = []
    with open(jsonl_path, 'r', encoding='utf-8') as infile:
        for line, entropy in zip(infile, semantic_entropies):
            data = json.loads(line.strip())
            data['sentropy'] = entropy
            updated_data.append(data)
    
    with open(jsonl_path, 'w', encoding='utf-8') as outfile:
        for data in updated_data:
            outfile.write(json.dumps(data) + '\n')

    print(f"SE: {semantic_entropies}")

