from tqdm import tqdm, trange
from time import sleep
import json
import requests
import urllib.parse
from redis import Redis


s2_api_key = 'y3OGRrc8mo7xNDW904Q9x5KqeffKeE3ftuOFWLu3'
#sch = SemanticScholar(api_key=s2_api_key)

redis = Redis(decode_responses=True, port=6380)


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


def search_papers(keyword_query, total_limit=None, secret =""):
    api_root = "https://partner.semanticscholar.org/graph/v1"
    full_results = []
    paper_index = set()
    
    query = f"{api_root}/paper/search?query={urllib.parse.quote(keyword_query)}&fields=paperId,url,title,abstract,venue,citationCount,fieldsOfStudy,year&limit=100"
    headers = {}
    if secret is not None and len(secret) > 0:
        headers['x-api-key'] = s2_api_key
    result = get(query, headers=headers)
    parsed = json.loads(result)
    
    total = parsed['total']
    if total_limit is None:
        total_limit = total
    
    for paper in parsed['data']:
        if paper['paperId'] not in paper_index:
            paper_index.add(paper['paperId'])
            full_results.append(paper)
    min_total = min(min(total_limit, total), 10000)
    bar = tqdm(total=min_total)
    offset=100
    while offset < min_total:
        sleep(0.001)
        result = get(f"{query}&offset={offset}", headers=headers)
        try:
            parsed = json.loads(result)
            if 'data' not in parsed:
                print(parsed)
            for paper in parsed['data']:
                if paper['paperId'] not in paper_index:
                    paper_index.add(paper['paperId'])
                    full_results.append(paper)
            offset+=100
            bar.update(100)
        except Exception as ignored:
            print(result, offset)
            bar.update(100)
            offset+=100
            continue
        
    return full_results


# def result_to_set(results, limit=None):
#     final = set()
#     index = 0
#     for result in tqdm(results, total=results.total):
#         final.add(result)
#         sleep(0.001)
#         if limit is not None and index == limit:
#           break
#         index+=1
#     return final

def extract_and_save_keyword(keyword, kwtype, limit=None, secret=""):
    full_result = search_papers(keyword, limit, secret)
    with open(kwtype+"_kw_"+keyword.replace(" ", "_")+".json", "w") as out_file:
        json.dump(full_result, out_file, indent=2)

nlp_keywords = ["Natural Language Processing", "NLP", "Text processing", "Human Language Technology", "Information extraction", "Knowledge Extraction", "Natural Language Understanding", "Natural Language Generation", "Natural Language Engineering", "Text generation", "Natural Language Learning", "Machine Translation", "Word Sense Disambiguation"]
ld_keywords = ["Ontology Engineering", "Linked Data", "Linguistic Linked Data", "Semantic Web", "RDF", "SKOS", "Ontolex", "Ontology"]




print("Processing NLP Keywords")
for kw in nlp_keywords:
    for kw2 in ld_keywords:
        print("Running search for "+kw+" "+kw2)
        extract_and_save_keyword(kw+", "+kw2, "comb", secret=s2_api_key)

# print("Processing NLP Keywords")
# for kw in ld_keywords:
#   print("Running search for "+kw)
#   extract_and_save_keyword(kw, "ld", secret=s2_api_key)
