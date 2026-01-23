import os
import json
import torch
import argparse 
import pytrec_eval
from typing import Dict
from models.load_models import load_model

os.environ["TOKENIZERS_PARALLELISM"] = "false"

metrics_dict = {
    'edir' : {"recall_1", "recall_3"},
    'cirr_val' : {"recall_1", "recall_5", "recall_10"},
    'fashioniq' : {"recall_10", "recall_50"},
    'circo_val' : {"map_cut"},
    'circo_test' : {"map_cut"},
    'test' : {'recall_1', 'recall_3'}
}

def load_dataset(dataset_name: str = "test"): 
    with open(f"dataset/{dataset_name}/queries.jsonl", "r") as f:
        queries = [json.loads(line) for line in f]

    with open(f"dataset/{dataset_name}/corpus.jsonl", "r") as f:
        corpus = [json.loads(line) for line in f]
    
    with open(f"dataset/{dataset_name}/instances.jsonl", "r") as f:
        instances = [json.loads(line) for line in f]
    
    qrels = {}
    for item in instances: 
        qrels[item['qid']] = {}
        for pos in item['pos']:
            qrels[item['qid']][pos] = 1

    return queries, corpus, qrels

def trec_eval(dataset_name: str, predictions: Dict, qrels: Dict):
    
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, metrics_dict[dataset_name])
    return evaluator.evaluate(predictions)

query_instruction = "Given an image, find a similar everyday image with the described changes: "
corpus_instruction = "Represent the given image. "

def main(args):
    model = load_model(args.model_id, args.model_name_or_path, device='cuda')
    queries, corpus, qrels = load_dataset(args.dataset)
    image_root_path = os.path.join(args.dataset_path, args.dataset)
    query_embs = model.query_emb(texts=[query['text'] for query in queries], images=[os.path.join(image_root_path, query['image']) for query in queries], instruction=query_instruction)
    corpus_embs = model.corpus_emb(images=[os.path.join(image_root_path, c['image']) for c in corpus], instruction=corpus_instruction)
    scores_mat = query_embs @ corpus_embs.T 
    scores_mat = scores_mat.to(torch.float32).cpu()
    # save the top-100 scores for each query 
    # save to {'query_id': {'corpus_id' : score, "corpus_id" : score, ...}, ...}"}}
    corpus_ids = [c['id'] for c in corpus]
    predictions = {}
    for i, query in enumerate(queries):
        # Get scores for this query
        query_scores = scores_mat[i]
        # Get top-100 indices and scores
        top_k_indices = top_k_indices = query_scores.argsort(descending=True)[:100]
        predictions[query['id']] = {
            corpus_ids[idx] : float(query_scores[idx]) 
            for idx in top_k_indices
        }

    results = trec_eval(args.dataset, predictions, qrels)

    os.makedirs(f"outputs/{args.model_id}", exist_ok=True)
    os.makedirs(f"outputs/{args.model_id}/results", exist_ok=True)
    if args.save_predictions:
        with open(f"outputs/{args.model_id}/{args.dataset}_predictions.json", "w") as f:
            json.dump(predictions, f)
    
    with open(f"outputs/{args.model_id}/results/{args.dataset}_results.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default='edir', choices=['edir', 'cirr_val', 'fashioniq', 'circo_val', 'cirr_test', 'circo_test', 'test'])
    parser.add_argument('--dataset_path', type=str, default='images/')
    parser.add_argument('--model_id', type=str)
    parser.add_argument('--model_name_or_path', type=str)   
    parser.add_argument('--save_predictions', type=bool, default=True)
    args = parser.parse_args()
    main(args)