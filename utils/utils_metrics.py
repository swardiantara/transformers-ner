from collections import defaultdict
from unittest import result
import numpy as np
from seqeval.metrics import classification_report

def get_entities(seq):
    """Gets entities from sequence.
    note: BIO
    Args:
        seq (list): sequence of labels.
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        seq = ['B-PER', 'I-PER', 'O', 'B-LOC', 'I-PER']
        get_entity_bio(seq)
        #output
        [['PER', 0,1], ['LOC', 3, 3], ['PER', 4, 4]]
    """
    if any(isinstance(s, list) for s in seq):
        seq = [item for sublist in seq for item in sublist + ['O']]

    prev_tag = 'O'
    prev_type = ''
    begin_offset = 0
    chunks = []
    for i, chunk in enumerate(seq + ['O']):
        tag = chunk[0]
        type_ = chunk.split('-')[-1]

        if end_of_chunk(prev_tag, tag, prev_type, type_):
            chunks.append((prev_type, begin_offset, i - 1))
        if start_of_chunk(prev_tag, tag, prev_type, type_):
            begin_offset = i
        prev_tag = tag
        prev_type = type_

    return set(chunks)


def end_of_chunk(prev_tag, tag, prev_type, type_):
    """Checks if a chunk ended between the previous and current word.

    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.

    Returns:
        chunk_end: boolean.
    """
    chunk_end = False

    if prev_tag == 'E': chunk_end = True
    if prev_tag == 'S': chunk_end = True

    if prev_tag == 'B' and tag == 'B': chunk_end = True
    if prev_tag == 'B' and tag == 'S': chunk_end = True
    if prev_tag == 'B' and tag == 'O': chunk_end = True
    if prev_tag == 'I' and tag == 'B': chunk_end = True
    if prev_tag == 'I' and tag == 'S': chunk_end = True
    if prev_tag == 'I' and tag == 'O': chunk_end = True

    if prev_tag != 'O' and prev_tag != '.' and prev_type != type_:
        chunk_end = True

    return chunk_end


def start_of_chunk(prev_tag, tag, prev_type, type_):
    """Checks if a chunk started between the previous and current word.

    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.

    Returns:
        chunk_start: boolean.
    """
    chunk_start = False

    if tag == 'B': chunk_start = True
    if tag == 'S': chunk_start = True

    if prev_tag == 'E' and tag == 'E': chunk_start = True
    if prev_tag == 'E' and tag == 'I': chunk_start = True
    if prev_tag == 'S' and tag == 'E': chunk_start = True
    if prev_tag == 'S' and tag == 'I': chunk_start = True
    if prev_tag == 'O' and tag == 'E': chunk_start = True
    if prev_tag == 'O' and tag == 'I': chunk_start = True

    if tag != 'O' and tag != '.' and prev_type != type_:
        chunk_start = True

    return chunk_start


def get_entities_bios(seq):
    """Gets entities from sequence.
    note: BIOS
    Args:
        seq (list): sequence of labels.
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        # >>> seq = ['B-PER', 'I-PER', 'O', 'S-LOC']
        # >>> get_entity_bios(seq)
        [['PER', 0,1], ['LOC', 3, 3]]
    """
    if any(isinstance(s, list) for s in seq):
        seq = [item for sublist in seq for item in sublist + ['O']]
    chunks = []
    chunk = [-1, -1, -1]
    for indx, tag in enumerate(seq):
        if tag.startswith("S-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[2] = indx
            chunk[0] = tag.split('-')[1]
            chunks.append(chunk)
            chunk = (-1, -1, -1)
        if tag.startswith("B-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[0] = tag.split('-')[1]
        elif tag.startswith('I-') and chunk[1] != -1:
            _type = tag.split('-')[1]
            if _type == chunk[0]:
                chunk[2] = indx
            if indx == len(seq) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
    return set([tuple(chunk) for chunk in chunks])


def get_entities_bio(seq):
    """Gets entities from sequence.
    note: BIO
    Args:
        seq (list): sequence of labels.
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        seq = ['B-PER', 'I-PER', 'O', 'B-LOC', 'I-PER']
        get_entity_bio(seq)
        #output
        [['PER', 0,1], ['LOC', 3, 3]]
    """
    if any(isinstance(s, list) for s in seq):
        seq = [item for sublist in seq for item in sublist + ['O']]
    chunks = []
    chunk = [-1, -1, -1]
    for indx, tag in enumerate(seq):
        if tag.startswith("B-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[0] = tag.split('-')[1]
            chunk[2] = indx
            if indx == len(seq) - 1:
                chunks.append(chunk)
        elif tag.startswith('I-') and chunk[1] != -1:
            _type = tag.split('-')[1]
            if _type == chunk[0]:
                chunk[2] = indx

            if indx == len(seq) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
    return set([tuple(chunk) for chunk in chunks])


def get_entities_span(starts, ends):
    if any(isinstance(s, list) for s in starts):
        starts = [item for sublist in starts for item in sublist + ['<SEP>']]
    if any(isinstance(s, list) for s in ends):
        ends = [item for sublist in ends for item in sublist + ['<SEP>']]
    chunks = []
    for start_index, start in enumerate(starts):
        if start in ['O', '<SEP>']:
            continue
        for end_index, end in enumerate(ends[start_index:]):
            if start == end:
                chunks.append((start, start_index, start_index + end_index))
                break
            elif end == '<SEP>':
                break
    return set(chunks)


def f1_score(true_entities, pred_entities):
    """Compute the F1 score."""
    nb_correct = len(true_entities & pred_entities)
    nb_pred = len(pred_entities)
    nb_true = len(true_entities)

    p = nb_correct / nb_pred if nb_pred > 0 else 0
    r = nb_correct / nb_true if nb_true > 0 else 0
    score = 2 * p * r / (p + r) if p + r > 0 else 0

    return score


def precision_score(true_entities, pred_entities):
    """Compute the precision."""
    nb_correct = len(true_entities & pred_entities)
    nb_pred = len(pred_entities)

    score = nb_correct / nb_pred if nb_pred > 0 else 0

    return score


def recall_score(true_entities, pred_entities):
    """Compute the recall."""
    nb_correct = len(true_entities & pred_entities)
    nb_true = len(true_entities)

    score = nb_correct / nb_true if nb_true > 0 else 0

    return score


def classification_report(true_entities, pred_entities, digits=5):
    """Build a text report showing the main classification metrics."""
    name_width = 0
    d1 = defaultdict(set)
    d2 = defaultdict(set)
    for e in true_entities:
        d1[e[0]].add((e[1], e[2]))
        name_width = max(name_width, len(e[0]))
    for e in pred_entities:
        d2[e[0]].add((e[1], e[2]))

    last_line_heading = 'macro avg'
    width = max(name_width, len(last_line_heading), digits)

    headers = ["precision", "recall", "f1-score", "support"]
    head_fmt = u'{:>{width}s} ' + u' {:>9}' * len(headers)
    report = head_fmt.format(u'', *headers, width=width)
    report += u'\n\n'

    row_fmt = u'{:>{width}s} ' + u' {:>9.{digits}f}' * 3 + u' {:>9}\n'

    ps, rs, f1s, s = [], [], [], []
    for type_name, type_true_entities in d1.items():
        type_pred_entities = d2[type_name]
        nb_correct = len(type_true_entities & type_pred_entities)
        nb_pred = len(type_pred_entities)
        nb_true = len(type_true_entities)

        p = nb_correct / nb_pred if nb_pred > 0 else 0
        r = nb_correct / nb_true if nb_true > 0 else 0
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0

        report += row_fmt.format(*[type_name, p, r, f1, nb_true], width=width, digits=digits)

        ps.append(p)
        rs.append(r)
        f1s.append(f1)
        s.append(nb_true)

    report += u'\n'

    # compute averages
    report += row_fmt.format('micro avg',
                             precision_score(true_entities, pred_entities),
                             recall_score(true_entities, pred_entities),
                             f1_score(true_entities, pred_entities),
                             np.sum(s),
                             width=width, digits=digits)
    report += row_fmt.format(last_line_heading,
                             np.average(ps, weights=s),
                             np.average(rs, weights=s),
                             np.average(f1s, weights=s),
                             np.sum(s),
                             width=width, digits=digits)

    return report


def convert_span_to_bio(starts, ends):
    labels = []
    for start, end in zip(starts, ends):
        entities = get_entities_span(start, end)
        label = ['O'] * len(start)
        for entity in entities:
            label[entity[1]] = 'B-{}'.format(entity[0])
            label[entity[1] + 1: entity[2] + 1] = ['I-{}'.format(entity[0])] * (entity[2] - entity[1])
        labels.append(label)
    return labels


def evaluation_table(flattened_true_list, flattened_pred_list):
    # Build the results object dynamically from unique list of flattened_true list value
    # Take the intersection between the true and pred list
    results = {
        'O' : {
            'tp': 0,
            'fp': 0,
            'fn': 0,
            'support': 0
        },
        'B-ACT' : {
            'tp': 0,
            'fp': 0,
            'fn': 0,
            'support': 0
        },
        'I-ACT' : {
            'tp': 0,
            'fp': 0,
            'fn': 0,
            'support': 0
        },
        'B-CMP' : {
            'tp': 0,
            'fp': 0,
            'fn': 0,
            'support': 0
        },
        'I-CMP' : {
            'tp': 0,
            'fp': 0,
            'fn': 0,
            'support': 0
        },
        'B-FNC' : {
            'tp': 0,
            'fp': 0,
            'fn': 0,
            'support': 0
        },
        'I-FNC' : {
            'tp': 0,
            'fp': 0,
            'fn': 0,
            'support': 0
        },
        'B-ISS' : {
            'tp': 0,
            'fp': 0,
            'fn': 0,
            'support': 0
        },
        'I-ISS' : {
            'tp': 0,
            'fp': 0,
            'fn': 0,
            'support': 0
        },
        'B-STE' : {
            'tp': 0,
            'fp': 0,
            'fn': 0,
            'support': 0
        },
        'I-STE' : {
            'tp': 0,
            'fp': 0,
            'fn': 0,
            'support': 0
        }
    }
    # i = 1
    # j = 1
    # for true in flattened_true_list:
    #         for pred in flattened_pred_list:
    #             if (i == j and true == pred): results[true]['tp'] += 1
    #             elif (i != j and true != pred): 
    #                 results[true]['fn'] += 1
    #                 results[pred]['fp'] += 1
    #             j += 1
    #         results[true]['support'] += 1
    #         i += 1

    for i in range (0, len(flattened_true_list)):
        if(flattened_true_list[i] == flattened_pred_list[i]):
            # true == pred
            results[flattened_true_list[i]]['tp'] += 1
        elif(flattened_true_list[i] != flattened_pred_list[i]):
            # true != pred
            results[flattened_true_list[i]]['fn'] += 1
            results[flattened_pred_list[i]]['fp'] += 1
        results[flattened_true_list[i]]['support'] += 1

    return results


def evaluation_score_model(evaluation_table):
    evaluation_score = {
        'O' : {
            'precision': 0,
            'recall': 0,
            'f1': 0
        },
        'B-ACT' : {
            'precision': 0,
            'recall': 0,
            'f1': 0
        },
        'I-ACT' : {
            'precision': 0,
            'recall': 0,
            'f1': 0
        },
        'B-CMP' : {
            'precision': 0,
            'recall': 0,
            'f1': 0
        },
        'I-CMP' : {
            'precision': 0,
            'recall': 0,
            'f1': 0
        },
        'B-f1C' : {
            'precision': 0,
            'recall': 0,
            'f1': 0
        },
        'I-f1C' : {
            'precision': 0,
            'recall': 0,
            'f1': 0
        },
        'B-ISS' : {
            'precision': 0,
            'recall': 0,
            'f1': 0
        },
        'I-ISS' : {
            'precision': 0,
            'recall': 0,
            'f1': 0
        },
        'B-STE' : {
            'precision': 0,
            'recall': 0,
            'f1': 0
        },
        'I-STE' : {
            'precision': 0,
            'recall': 0,
            'f1': 0
        },
        'micro_avg': {
            'precision': 0,
            'recall': 0,
            'f1': 0
        },
        'macro_avg': {
            'precision': 0,
            'recall': 0,
            'f1': 0
        },
        'accuracy': 0
    }

    sum = {
        'tp': 0,
        'fp': 0,
        'fn': 0,
        'precision': 0,
        'recall': 0,
        'f1': 0,
        'support': 0
    }

    for tag in evaluation_table:
        # Compute per class evaluation score
        precision = tag['tp'] / (tag['tp'] + tag['fp'])
        recall = tag['tp'] / (tag['tp'] + tag['fn'])
        f1 = (2* (precision*recall)) / (precision + recall)

        evaluation_score[tag]['precision'] = precision
        evaluation_score[tag]['recall'] = recall
        evaluation_score[tag]['f1'] = f1

        # Store the sum of per class TP, FP, FN for computing micro_avg
        sum['tp'] = sum['tp'] + tag['tp']
        sum['fp'] = sum['fp'] + tag['fp']
        sum['fn'] = sum['fn'] + tag['fn']

        # Store the sum of per class evaluation score (precision, recall and f1) computing for macro_avg
        sum['precision'] = sum['precision'] + precision
        sum['recall'] = sum['recall'] + recall
        sum['f1'] = sum['f1'] + f1

        # Store the sum of support for computing accuracy
        sum['support'] = sum['support'] + tag['support']
    
    # Compute the macro_avg evaluation score
    evaluation_score['macro_avg']['precision'] = sum['precision'] / len (evaluation_table)
    evaluation_score['macro_avg']['recall'] = sum['recall'] / len (evaluation_table)
    evaluation_score['macro_avg']['f1'] = sum['f1'] / len (evaluation_table)
        
    # Compute the micro_avg evaluation score
    evaluation_score['micro_avg']['precision'] = sum['tp'] / (sum['tp'] + sum['fp'])
    evaluation_score['micro_avg']['precision'] = sum['tp'] / (sum['tp'] + sum['fn'])
    evaluation_score['micro_avg']['f1'] = sum['tp'] / (sum['tp'] + ((sum['fp'] + sum['fn']) / 2))
    evaluation_score['accuracy'] = sum['tp'] / sum['support']

    return evaluation_score
# starts = [['O', 'O', 'O', 'MISC', 'O', 'O', 'O'], ['PER', 'O', 'O']]
# ends = [['O', 'O', 'O', 'O', 'O', 'MISC', 'O'], ['O', 'PER', 'O']]
#
# print(convert_span_to_bio(starts, ends))
