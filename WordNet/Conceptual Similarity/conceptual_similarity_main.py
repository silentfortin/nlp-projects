import numpy as np
import pandas as pd
from nltk.corpus import wordnet as wn
from scipy.sparse.csgraph import shortest_path
from scipy.stats import pearsonr
from scipy.stats import spearmanr

depth_max = 20

def read_from_csv(csv_path):
    if len(csv_path) == 0:
        print("\n Path is empty")
        exit()
    pair = pd.read_csv(csv_path)
    word1 = pair['Word 1']
    word2 = pair['Word 2']
    human_pair = pair['Human (mean)']
    row_number = len(word1.iloc[:])

    return word1, word2, human_pair, row_number


def get_max_similarity(t1, t2, sim_metric):
    """
    calculates the maximum similarity between two terms using a specific similarity metric
    Args:
        t1: the first term to compare
        t2: the second term to compare
        sim_metric: the similarity metric to use. Can be one of ['wupalmer','shortest_path','lac']
    Returns:
        a tuple containing the maximum similarity score and the pair of synsets that achieved it
    """
    m_similarity = 0
    max_synsets = ()

    # get all the synsets for each term
    synset1 = wn.synsets(t1)
    synset2 = wn.synsets(t2)

    # compare each synset from term 1 with each synset from term 2
    for s1 in synset1:
        for s2 in synset2:
            # compute the similarity using the specified metric
            if sim_metric == 'wupalmer':
                computed_similarity = wu_and_palmer(s1, s2)
            elif sim_metric == 'shortest_path':
                computed_similarity = shortest_path_dist(s1, s2)
            elif sim_metric == 'lac':
                computed_similarity = leacock_and_chodorow(s1, s2)
            else:
                raise ValueError(
                    "Incorrect usage of similarity metrics parameter\nUse one of ['wupalmer','shortest_path',"
                    "'lac']")
            # keep track of the maximum similarity and the synsets that achieved it
            if computed_similarity > m_similarity:
                m_similarity = computed_similarity
                max_synsets = (s1, s2)

    return m_similarity, max_synsets


def calculate_similarity(similarity_metric, n_rows, t1, t2, hum):
    """
    calculates similarity between pairs of terms using a given similarity metric
    Args:
        similarity_metric: the name of the similarity metric to use
        n_rows: the number of rows to process
        t1:the first term in each pair
        t2:the second term in each pair
        hum:human ratings of similarity for each pair
    Returns: None
    """
    # initialize an empty list to store the results
    similarity_result = []

    # loop through each row
    for i in range(n_rows):
        # get the terms and human rating for the current row
        term1 = t1.iloc[i]
        term2 = t2.iloc[i]
        human = hum.iloc[i]

        # calculate the maximum similarity between the terms using the given metric
        m_similarity = get_max_similarity(term1, term2, similarity_metric)[0]

        # append the result to the similarity_result list and to a tuple for each pair
        similarity_result.append(m_similarity)
        similarity.append((term1, term2, m_similarity, human))

    return similarity_result, similarity

def depth(synset):
    """
    calculate the depth of a given synset in the WordNet hierarchy
    Args:
        synset: a synset object
    Returns:
        the depth of the synset in the WordNet hierarchy
    """
    # if the synset is None, return 0
    if synset is None:
        return 0

    # initialize the list of synsets to be explored with the given synset
    to_be_explored = [synset]

    # initialize the computed depth to 0
    computed_depth = 0

    # while there are synsets to be explored
    while to_be_explored:

        # get the next synset to explore
        current_synset = to_be_explored[0]

        # remove the current synset from the list of synsets to be explored
        to_be_explored = to_be_explored[1:]

        # get the hypernyms of the current synset
        hypernyms = current_synset.hypernyms()

        # if there are hypernyms, add them to the list of synsets to be explored
        # and increment the computed depth by 1
        if hypernyms:
            to_be_explored += hypernyms
            computed_depth += 1

    # return the computed depth
    return computed_depth


def hypernym_paths(synset):
    """
    given a synset, compute all its hypernyms and map them to
    the length of the path that connects them to the synset
    :param synset: the synset whose hypernyms are to be computed
    :return: a list containing three elements:
        - a set of the hypernym names
        - a list of the hypernyms
        - the length of the path from the synset to the root of WordNet
    """
    hypernyms_map = {synset.name(): 0}
    hypernyms = [synset]
    to_be_explored = [synset]
    computed_depth = 1

    for current_synset in to_be_explored:
        hypernyms_of_current_synset = current_synset.hypernyms()

        for hypernym in hypernyms_of_current_synset:
            name = hypernym.name()

            if name not in hypernyms_map:
                hypernyms_map[name] = computed_depth
                hypernyms.append(hypernym)
                to_be_explored.append(hypernym)

        computed_depth += 1

    return [hypernyms_map, hypernyms, computed_depth - 1]


def find_lcs(synset1, synset2):
    """
    computes the least common subsumer for the given synsets
    :param synset1: the first synset
    :param synset2: the second synset

    :return: the least common subsumer of synset1 and synset2
    """

    # compute the hypernym paths for each synset
    map1, hy1, depth1 = hypernym_paths(synset1)
    map2, hy2, depth2 = hypernym_paths(synset2)

    # find the first common hypernym
    for hypernym in hy2:
        if hypernym.name() in map1:
            return hypernym


# SIMILARITY METRICS
def wu_and_palmer(first_sense, second_sense):
    """
    this function calculates the Wu & Palmer similarity metric between two WordNet senses
    :param first_sense: the first WordNet sense
    :param second_sense: the second WordNet sense
    :return: the Wu & Palmer similarity between the two senses
    """

    # get the least common subsumer (LCS) of sense1 and sense2
    lcs = find_lcs(first_sense, second_sense)

    # calculate the depth of the LCS and the depths of the two senses
    depth_lcs = depth(lcs)
    depth_sense1 = depth(first_sense)
    depth_sense2 = depth(second_sense)

    # calculate the Wu & Palmer similarity
    if depth_sense1 == 0 or depth_sense2 == 0:
        return 0
    else:
        similarity = 2 * depth_lcs / (depth_sense1 + depth_sense2)

    return round(similarity * 10, 2)


"""
2 - Shortest Path

computes the semantic similarity of two concepts by calculating the shortest
path between them in taxonomy. The intuition behind the algorithm is
that the shorter the path between concepts in a hierarchy
the more similar they are.

sim_path(s1, s2) = 2 * depthMAX - len(s1, s2)

len(s1, s2) = shortest path between concepts
depthMax = a fixed value that refers to the maximum depth of the taxonomy (20)
"""


def shortest_path(synset1, synset2):
    """
    computes the shortest path between two synsets by finding their closest common hypernym
    :param synset1: The first synset
    :param synset2: The second synset
    :return: the length of the shortest path between the two synsets
    """
    # compute the hypernym paths and maps for both synsets
    synset1_map, synset1_hypernyms, synset1_depth = hypernym_paths(synset1)
    synset2_map, synset2_hypernyms, synset2_depth = hypernym_paths(synset2)

    # iterate over the hypernyms of the second synset
    for hypernym in synset2_hypernyms:
        hypernym_name = hypernym.name()
        # if a hypernym is found in the hypernym map of the first synset,
        # calculate the shortest path between the two synsets
        if hypernym_name in synset1_map:
            synset1_to_common = synset1_map[hypernym_name]
            synset2_to_common = synset2_map[hypernym_name]
            return synset1_to_common + synset2_to_common

    # if no common hypernym is found, return the maximum depth
    return 2 * depth_max


def shortest_path_dist(synset1, synset2):
    """
    computes the semantic similarity of two concepts by calculating the shortest
    path between them. The intuition behind the algorithm is
    that the shorter the path between concepts in a hierarchy, the more similar they are.

    formula: sim_path(s1, s2) = 2 * depth_max - len(s1, s2)
        - len(s1, s2): shortest path between concepts
        - depth_max: maximum depth of the taxonomy (20)

    :param synset1: a synset in WordNet
    :param synset2: another synset in WordNet
    :return: the shortest path similarity between synset1 and synset2
    """
    shortest_path_len = shortest_path(synset1, synset2)
    similarity = 2 * depth_max - shortest_path_len
    return similarity


def leacock_and_chodorow(synset1, synset2):
    """
    computes the Leacock & Chodorow similarity

    :param c1: The first concept
    :param c2: The second concept
    :return: the similarity score between c1 and c2
    """
    path_len = shortest_path(synset1, synset2) + 1
    denominator = 2 * (depth_max + 1)
    similarity = -np.log(path_len / denominator)

    return round(similarity, 2)



def get_correlation_indices(correlation_index, read_human_pair):
    """
    computes correlation indices which can be Spearman or Pearson
    we'll make a comparison between similarities in wordSin353.csv and the found ones
    :param correlation_index:
    :param read_human_pair:
    :return:
    """
    correlation = 0
    if correlation_index == 'spearman':
        correlation = spearmanr(read_human_pair, similarity_result)[0]
    elif correlation_index == 'pearson':
        correlation = pearsonr(read_human_pair, similarity_result)[0]

    return correlation


def get_similarity():
    return similarity


# RUN
similarity = []
similarity_result = []

csv_path = '../wordSin353.csv'
# SIMILARITY can be [wupalmer, shortest_path or v]
similarity_metrics = 'lac'

# reading from CSV
if csv_path is not None:
    word1, word2, human_pair, row_number = read_from_csv(csv_path)

similarity_result, similarity = calculate_similarity(similarity_metrics, row_number, word1, word2, human_pair)


pears = round(get_correlation_indices('pearson', human_pair) * 100, 2)
spear = round(get_correlation_indices('spearman', human_pair) * 100, 2)

for el in get_similarity():
    print("\nWords: ", el[0] + " - " + el[1])
    print("computed similarity: ", el[2])
    print("wordSin353 similarity: ", el[3])

print("\n" + "pearson correlation : " + str(pears))
print("spearman correlation : " + str(spear))
print("\n" + "Similarity metric: ", similarity_metrics)
