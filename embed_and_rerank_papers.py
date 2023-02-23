from argparse import ArgumentParser
import json
from pathlib import Path
from pydoc import describe
import sys

import torch

from tqdm import tqdm, trange
import numpy as np


parser = ArgumentParser()

parser.add_argument(
    "input_file",
    type=Path,
    help="Input json file resulting from SemanticScholar extraction")

parser.add_argument("--model",
                    "-m",
                    nargs=1,
                    default=["bigscience/bloom-560m"],
                    help="HF Hub Model to use. Default: bloom-560m")

parser.add_argument("--device",
                    "-d",
                    nargs=1,
                    default=["cpu"],
                    help="Compute device to use. Default: cpu")

parser.add_argument(
    "--limit",
    "-l",
    nargs=1,
    type=int,
    help=
    "Apply a limit to the number of paper (first K papers in decreasing relevance score order)"
)

parser.add_argument("--threshold",
                    "-t",
                    nargs=1,
                    type=float,
                    help="Apply a threshold filter to the papers.")

parser.add_argument("--batch_size",
                    nargs=1,
                    type=int,
                    default=[1],
                    help="Batch size for the embedding step. Default: 1")

DEFAULT_PROMPT = """Include Natural Language Processing (NLP) papers that make use of Linked Data as an integral part of the method or system,
 where the system or method is explicitly designed to take advantage of Linked Data capabilities such as dynamicity, interoperability, 
 reasoning, structure conditioning, etc. Also include NLP papers that use Linked Data to describe or specify a pipelines, tasks, or 
 processes to facilitate interoperability. Please EXCLUDE NLP papers that just use Linked Data as a resource or just produce Linked Data 
 as their output."""
parser.add_argument(
    "--prompt",
    "-p",
    nargs=1,
    type=str,
    default=[DEFAULT_PROMPT],
    help=
    "The prompt is used to rerank papers vectors by similarity to promp when --promp_rerank is activated"
)

parser.add_argument("--prompt_rerank", "-r", action='store_true', help="If set, reranks papers according to prompt")

parser.add_argument("--rerank_dist", "-ds", nargs=1, type=str, default=["L2"], help="Distance fuction tu use for reranking. L2 or cos. Default: L2")


def hf_embedding(flat_papers,
                 model='bigscience/bloom-560m',
                 include_fields=None,
                 batch_size=10, device="cpu"):
    import torch

    if include_fields is None:
        include_fields = ['title', 'abstract', 'summary']

    paper_strings = []
    scores = []
    for paper in tqdm(flat_papers):
        final_str = ""
        for field in include_fields:
            if field in paper:
                final_str += paper[field] + "\n"

        paper_strings.append(final_str)
        scores.append(paper['relevance_score'])
    scores = np.array(scores)
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModel.from_pretrained(model)
    model = model.to(device)

    embeddings = []

    for i in trange(0, len(paper_strings), batch_size):
        with torch.no_grad():
            inputs = tokenizer(paper_strings[i:i + batch_size],
                               return_tensors="pt",
                               padding=True,
                               truncation=True,
                               max_length=512).to(device)
            vect = model(
                **inputs).last_hidden_state.detach().cpu().numpy()[:, 0, :]
        print(vect.shape)
        embeddings.append(vect)
    embeddings = np.concatenate(embeddings)
    return embeddings, scores


def hf_prompt_embedding(prompt, model='bigscience/bloom-560m', device="cpu"):
    import torch
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModel.from_pretrained(model)
    model = model.to(device)
    with torch.no_grad():
        inputs = tokenizer(prompt,
                           return_tensors="pt",
                           padding=True,
                           truncation=True,
                           max_length=512).to(device)
        vect = model(**inputs).last_hidden_state.detach().cpu().numpy()[:,
                                                                        0, :]
        return vect

def vec_cos(A, B):
    dot = np.einsum('ij,ij->i', A, B)
    norm_A = np.linalg.norm(A, axis=1)
    norm_B = np.linalg.norm(B, axis=1)
    return dot / norm_A*norm_B

def vec_eucl(A, B):
  a_min_b = A - B
  return np.sqrt(np.einsum('ij,ij->i', a_min_b, a_min_b))
  
if __name__ == "__main__":
    args = parser.parse_args()

    device = "cpu" if args.device[0] == "cpu" else (
    args.device[0] if torch.cuda.is_available() else "cpu")

    model = args.model[0]
    save_suffix = model.split("/")[1]

    with open(args.input_file) as f:
        papers = json.load(f)
    flat_papers = list(papers.values())
    flat_papers = sorted(flat_papers,
                         key=lambda x: x['relevance_score'],
                         reverse=True)
    if args.limit is not None:
        flat_papers[:args.limit[0]]
        save_suffix += f"_limit{args.limit[0]}"
    if args.threshold:
        save_suffix += f"_th{args.threshold[0]}"
        flat_papers = [
            paper for paper in flat_papers
            if paper['relevance_score'] > args.threshold[0]
        ]

    if Path(f"paper_embeddings_{save_suffix}.npy").exists():
        embedding = np.load(f"paper_embeddings_{save_suffix}.npy")
        scores = np.load(f"paper_scores_{save_suffix}.npy")
    else:
        embedding, scores = hf_embedding(flat_papers, model=model, batch_size=args.batch_size[0], device=device)
        np.save(f"paper_embeddings_{save_suffix}.npy", embedding)
        np.save(f"paper_scores_{save_suffix}.npy", embedding)


    if args.prompt_rerank:
        
        if Path(f"paper_prompt_{save_suffix}.npy").exists():
            prompt_vector = np.load(f"paper_prompt_{save_suffix}.npy")
        else:
            prompt_vector = hf_prompt_embedding(args.prompt[0], model=model)
            np.save(f"paper_prompt_{save_suffix}.npy", prompt_vector)
        tiled_prompt = np.tile(prompt_vector, (embedding.shape[0], 1))

        if args.rerank_dist[0] == "L2":
            similarities = vec_eucl(tiled_prompt, embedding)
        else:
            similarities = vec_cos(tiled_prompt, embedding)
            permutation = np.flip(np.argsort(similarities))
            flat_papers = [flat_papers[index] for index in permutation]
            embedding = embedding[permutation]

        with open(f"papers_{save_suffix}.json", "w") as f:
            json.dump(flat_papers, f)

        save_suffix +="_rank_permutation"
        np.save(f"paper_embeddings_{save_suffix}.npy", embedding)
        np.save(f"paper_scores_{save_suffix}.npy", embedding)

    
