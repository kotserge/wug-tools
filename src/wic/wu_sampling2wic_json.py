import json
import argparse

import numpy as np
import pandas as pd


def sampled_uses_json_generator(id_prefix: str, lang: str, path_file: str, path_out: str, sample_size: int, groups: list = None, include_group_info: bool = False):
    dfs = _generate_grouped_dataframes(path_file, groups)
    wu_pairs = _randomsample_groupings(dfs, sample_size)

    json_output = [_gen_dict_object('{}.{}'.format(
        id_prefix, i), lang, wu1, wu2, include_group_info) for i, (wu1, wu2) in enumerate(wu_pairs)]

    with open(path_out, 'w') as file:
        file.write(json.dumps(json_output, indent=4, ensure_ascii=False))
    file.close()


def _generate_grouped_dataframes(path_file: str, groups_to_use: list = None) -> list:
    df = pd.read_csv(path_file, delimiter='\t')
    # TODO: change df based on included, excluded groups
    if groups_to_use is not None:
        df_groupings = [df[df.grouping == x] for x in groups_to_use]
    else:
        df_groupings = [df[df.grouping == x] for x in df.grouping.unique()]

    return df_groupings


def _randomsample_groupings(dfs: list, sample_size: int):
    if len(dfs) == 0 or any([len(df) == 0 for df in dfs]):
        raise ValueError("Not enough uses to sample from")

    # Generate grouping combination
    grouping_combinations = [np.random.choice(
        len(dfs), size=2, replace=False) for _ in range(sample_size)]

    for group in grouping_combinations:
        sample1 = dict(dfs[group[0]].iloc[np.random.choice(
            len(dfs[group[0]].index))])
        sample2 = dict(dfs[group[1]].iloc[np.random.choice(
            len(dfs[group[1]].index))])

        yield sample1, sample2


def _gen_dict_object(id_: str, lang: str, sentence1: list, sentence2: list, include_group_info: bool = False) -> dict:

    _lookup_table_pos = {'german': _lookup_pos_maingroup_german,
                         'english': _lookup_pos_maingroup_english,
                         'swedish': _lookup_pos_maingroup_swedish}

    pos = sentence1['pos'] if sentence1['pos'] == sentence2['pos'] else ''
    pos = _lookup_table_pos[lang](pos)
    start1, end1 = sentence1['indexes_target_token'].split(':')
    start2, end2 = sentence2['indexes_target_token'].split(':')

    tmp = {"id": id_, "lemma": sentence1['lemma'], "pos": pos,
           "sentence1": sentence1['context'], "sentence2": sentence2['context'],
           "start1": str(start1), "end1": str(end1),
           "start2": str(start2), "end2": str(end2)}

    if include_group_info:
        tmp.update({"group1": str(sentence1['grouping']), "group2": str(sentence2['grouping'])})

    return tmp


def _clean_lemma(lemma: str) -> str:
    return lemma.split('_')[0].strip()


def _lookup_pos_maingroup_german(pos: str) -> str:
    if pos.lower().startswith('n'):
        return 'NOUN'
    if pos.lower().startswith('v'):
        return 'VERB'
    if pos.lower().startswith('adj'):
        return 'ADJ'
    if pos.lower().startswith('adv'):
        return 'ADV'
    return ''


def _lookup_pos_maingroup_english(pos: str) -> str:
    if pos.lower().startswith('n'):
        return 'NOUN'
    if pos.lower().startswith('v'):
        return 'VERB'
    if pos.lower().startswith('j'):
        return 'ADJ'
    if pos.lower().startswith('r'):
        return 'ADV'
    return ''


def _lookup_pos_maingroup_swedish(pos: str) -> str:
    return ''


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generates a JSON File randomsampled between groupings from a uses.csv')
    parser.add_argument('path_in', metavar='i', type=str, help='Path to uses.csv')
    parser.add_argument('path_out', metavar='o', type=str,
                        help='Path to file where the output will be written. If the file does not exit, a new file will be created. If it exist, it will be overwritten')
    parser.add_argument('id_prefix', metavar='id', type=str, help='Identifier to use')
    parser.add_argument('lang', metavar='lang', type=str, help='What language the uses.csv is from')
    parser.add_argument('sample_size', metavar='N', type=int, help='Number of samples to draw')
    parser.add_argument('-g', type=int, nargs='+', help='Specific groups to use. If empty, all groups will be used')
    parser.add_argument('--gi', action='store_true', help='If group info should be included in the JSON')
    args = parser.parse_args()
    sampled_uses_json_generator(args.id_prefix, args.lang, args.path_in, args.path_out, args.sample_size, args.g, args.gi)
