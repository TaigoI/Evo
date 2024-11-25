########################################################################################################################
#Imports
########################################################################################################################

from openai import OpenAI
from json import loads as json_loads
from json_repair import repair_json
from itertools import combinations

from nltk import download as nltk_download
nltk_download('punkt')
nltk_download('punkt_tab')
nltk_download('rslp')

from nltk import sent_tokenize as break_into_sentences
from nltk.stem import RSLPStemmer as stemmer

from re import search as re_search
import pandas as pd

import logging
import functools

########################################################################################################################
#Inference
########################################################################################################################

llm_client = OpenAI(
    api_key="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",
    base_url="http://177.53.19.28:8080/v1"  # "https://integrate.api.nvidia.com/v1"
)

MODEL = "model"  # "meta/llama-3.1-405b-instruct"


@functools.cache
def llm_inference(system_prompt: str, user_prompt: str):
    completion = llm_client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.7,
        top_p=0.7,
        max_tokens=2048,
        stream=False
    )

    return completion.choices[0].message.content

def extract_array(generated:str):
    return re_search(r"(\[[\W|\w|\s]*\])", generated).group(0)

########################################################################################################################
#Infer graph
########################################################################################################################

knowledge_graph_prompt = """
Considando a frase abaixo, responda somente com o JSON abaixo, adicionando quantas entradas forem necessárias:
[
    {"entidade_origem": "Nome da entidade", "relacionamento": "Uma única palavra, preferencialmente verbo, que descreve o relacionamento entre as entidades", "entidade_destino": "Nome da entidade"},
]
"""
def add_to_relationships(item: dict, relationships: list):
    origin_entity: list[str] = item["entidade_origem"]
    relationship: str = item["relacionamento"]
    destination_entity: list[str] = item["entidade_destino"]

    relationships.append((origin_entity, destination_entity, relationship))
@functools.cache
def infer_knowledge_graph(phrase: str) -> list[tuple[str, str, str]]:
    generated = extract_array(
        llm_inference(knowledge_graph_prompt, phrase)
    )

    logging.warning(generated)

    pair_relationships = []
    try:
        json_completion = json_loads(repair_json(generated))
        for item in json_completion:
            add_to_relationships(item, pair_relationships)

    except Exception as e:
        logging.error("FAILED TO GET GRAPH FROM COMPLETION")
        logging.error(e)

    return pair_relationships

########################################################################################################################
#Group named entities
########################################################################################################################

entity_grouping_prompt = """
Considere a lista de entidades que será fornecida abaixo. Seu objetivo é agrupar as entidades que se referem a uma mesma coisa, utilizando nomes diferentes.
Responda EXCLUSIVAMENTE com o JSON abaixo, onde a chave é o nome principal da entidade e o valor é uma lista de outras entidades que devem ser agrupadas com ela.
[
    {"entidade": [Outras entidades que devem ser agrupadas com esta]}
]
"""
@functools.cache
def infer_name_replacement(phrase: str) -> dict[str, str]:
    generated = extract_array(
        llm_inference(entity_grouping_prompt, phrase)
    )
    logging.warning(generated)

    to_replace = {}
    try:
        json_completion: list[dict[str, list[str]]] = json_loads(repair_json(generated))
        for item in json_completion:
            logging.warning(item)
            for main_entity, other_names in item.items():
                for other_name in other_names:
                    to_replace[other_name.lower().strip()] = main_entity.lower().strip()
    except Exception as e:
        logging.error("FAILED TO GET REPLACEMENTS FROM COMPLETION")
        logging.error(e)

    return to_replace

########################################################################################################################
#Text to graph
########################################################################################################################

def text_to_graph(text: str, should_break: bool = False) -> pd.DataFrame:
    connections = []
    sentences = [text]

    if should_break:
        sentences += break_into_sentences(text)

    for sentence in sentences:
        connections += infer_knowledge_graph(sentence)

    return pd.DataFrame(connections, columns=["entity1", "entity2", "relationship"]).apply(lambda x: x.str.lower())


def unify_graph_terms(graphs: list[pd.DataFrame], iterations: int = 2) -> list[pd.DataFrame]:
    all_named_entities = []
    replacements = {}

    for _ in range(iterations):
        all_named_entities = pd.concat([graph.melt()["value"] for graph in graphs]).unique()

        replacements = infer_name_replacement(str(all_named_entities))
        graphs = [graph.map(lambda x: replacements.get(x.lower(), x).lower()) for graph in graphs]
        replaced_named = {replacements.get(name, name) for name in all_named_entities}

    return graphs, replaced_named, replacements