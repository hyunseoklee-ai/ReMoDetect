import os.path
import random
import datasets

SEPARATOR = '<<<SEP>>>'


DATASETS = ['writing', 'english', 'german', 'pubmed']

def load_dataset(path, name=None, split=None, cache_dir=None):
    # use local model if it exists
    local_path = os.path.join(cache_dir, f'local.{path}_{name}_{split}')
    if os.path.exists(local_path):
        return datasets.load_from_disk(local_path)
    return datasets.load_dataset(path, name, split=split, cache_dir=cache_dir)

def load_pubmed(cache_dir):
    data = load_dataset('pubmed_qa', 'pqa_labeled', split='train', cache_dir=cache_dir)
    
    # combine question and long_answer
    data = [f'Question: {q} Answer:{SEPARATOR}{a}' for q, a in zip(data['question'], data['long_answer'])]

    return data


def process_prompt(prompt):
    return prompt.replace('[ WP ]', '').replace('[ OT ]', '')


def process_spaces(story):
    return story.replace(
        ' ,', ',').replace(
        ' .', '.').replace(
        ' ?', '?').replace(
        ' !', '!').replace(
        ' ;', ';').replace(
        ' \'', '\'').replace(
        ' ’ ', '\'').replace(
        ' :', ':').replace(
        '<newline>', '\n').replace(
        '`` ', '"').replace(
        ' \'\'', '"').replace(
        '\'\'', '"').replace(
        '.. ', '... ').replace(
        ' )', ')').replace(
        '( ', '(').replace(
        ' n\'t', 'n\'t').replace(
        ' i ', ' I ').replace(
        ' i\'', ' I\'').replace(
        '\\\'', '\'').replace(
        '\n ', '\n').strip()

def load_writing(cache_dir=None):
    import json
    with open(f'exp/data/writing_gpt-4.raw_data.json') as f:
        cur_json = json.load(f)
        # new_json = {'original':[],'sampled':[]}
        # new_json['original'] = cur_json['original']
    return cur_json['original']


def load(name, cache_dir, **kwargs):
    if name in DATASETS:
        load_fn = globals()[f'load_{name}']
        return load_fn(cache_dir=cache_dir, **kwargs)
    else:
        raise ValueError(f'Unknown dataset {name}')