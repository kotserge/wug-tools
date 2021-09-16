import os
import json


def input_json_generator(id_prefix: str, lang: str, path_file: str, path_out: str):
    wu_pairs = _gen_wu_pairs_from_file(path_file, lang)

    dict_objects = [_gen_dict_object('{}.{}'.format(
        id_prefix, i), wu1, wu2) for i, (wu1, wu2) in enumerate(wu_pairs)]

    with open(path_out, 'w') as file:
        file.write(json.dumps(dict_objects, indent=4, ensure_ascii=False))
    file.close()


def _gen_wu_pairs_from_file(path_file: str, lang: str):
    wu = []

    with open(path_file, 'r') as file:
        # skip header
        file.readline()
        line = file.readline()
        while line:
            line = line.split('\t')
            # lemma, pos, context start, end,
            wu.append([_clean_lemma(line[0]), _lookup_table_pos[lang](line[1]), line[6].strip(), *line[7].split(':')])
            line = file.readline()
    file.close()

    for i in range(len(wu)):
        for j in range(i + 1, len(wu)):
            yield wu[i], wu[j]


def _gen_dict_object(id_: str, sentence1: list, sentence2: list) -> dict:
    pos = sentence1[1] if sentence1[1] == sentence2[1] else ''
    return {"id": id_, "lemma": sentence1[0], "pos": pos,
            "sentence1": sentence1[2], "sentence2": sentence2[2],
            "start1": str(sentence1[3]), "end1": str(sentence1[4]),
            "start2": str(sentence2[3]), "end2": str(sentence2[4])}


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


_lookup_table_pos = {'german': _lookup_pos_maingroup_german,
                     'english': _lookup_pos_maingroup_english,
                     'swedish': _lookup_pos_maingroup_swedish}
