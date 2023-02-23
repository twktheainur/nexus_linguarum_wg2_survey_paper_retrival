from tqdm import tqdm, trange
from time import sleep
import json
import requests
import urllib.parse
from redis import Redis

import sys

s2_api_key = sys.argv[1]

redis = Redis(decode_responses=True, port=6379)


def get(url: str, headers, timeout= None):
    page_text = redis.get(url)
    try:
        if not page_text:
            result = requests.get(url, headers=headers, timeout=timeout)
            if result.status_code < 400:
                page_text = result.text
                if page_text is not None:
                    redis.set(url, page_text)
            else:
                print(result, result.text)
                return None
    except requests.exceptions.ReadTimeout:
        page_text = None
    except requests.exceptions.MissingSchema:
        page_text = None
    except requests.exceptions.ConnectTimeout:
        page_text = None
    return page_text

def fetch_paper(id, secret =""):
    api_root = "https://partner.semanticscholar.org/graph/v1"
    full_results = []
    paper_index = set()
    
    #&fields=embedding,tldr
    query = f"{api_root}/paper/{id}?fields=embedding,tldr,year,authors.name"
    headers = {}
    if secret is not None and len(secret) > 0:
        headers['x-api-key'] = s2_api_key
    
    result = get(query, headers=headers)
    try:
        parsed = json.loads(result)
    except: 
        print(id, result)
        return None
    return parsed

import json
papers = {}
with open('final_selection_t0.45.json') as f:
    papers = json.load(f)

    
final_selection = {paper['paperId']: paper for paper in papers}

counter=20
for paper_id in tqdm(final_selection):
    info = fetch_paper(paper_id, secret=s2_api_key)
    if info is not None:
        if 'embedding' in info:
            final_selection[paper_id]['embedding'] = info['embedding']['vector']
        else:
            final_selection[paper_id]['embedding']
        if 'tldr' in info and info['tldr'] is not None:
            final_selection[paper_id]['summary'] = info['tldr']['text']
        else:
            final_selection[paper_id]['summary'] = ""
    
        final_selection[paper_id]['year'] = info['year']
        final_selection[paper_id]['authors'] = ", ".join([ author['name'] for author in info['authors']])
    sleep(0.01)
    


with open('final_selection_t0.45_augmented.json', "w") as f:
    json.dump(final_selection, f)
