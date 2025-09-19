import collections
import math
from collections import defaultdict
import numpy as np
import time

# chosen language
language = 'en'
# transition_smoothing = 0.00000000000000000001
tag_order = {'O': 0, 'B-MISC': 1, 'I-MISC': 2, 'B-LOC': 3, 'I-LOC': 4, 'B-ORG': 5, 'I-ORG': 6, 'B-PER': 7, 'I-PER': 8}

# EPSILON is a small constant used on the probabilities before taking the logarithm to avoid
# 'RuntimeWarning: logarithm of zero or a negative number error'
EPSILON = 1e-9


def read_dataset(file_path):
    """
    This function opens the file and reads its contents into a list of strings,
    where each string represents a line in the file. If the file is empty, it raises a ValueError.
    Args:
        file_path: the file path of the dataset to be read
    Returns: a list of strings, where each string is a line from the dataset.
    """
    with open(file_path, encoding="utf8") as fileObj:
        dataset_lines = fileObj.readlines()
        if not dataset_lines:
            raise ValueError("Error, file is empty.")
        return dataset_lines


def compute_learning():
    """
    Starts the learning phase.

    Returns:
    - sentence_number: int, the number of sentences in the training data
    - tag_set: list of str, the unique tags in the training data
    - tags_occurrence: defaultdict of int, the number of times each tag occurs in the training data
    - final_tags_occurrence: defaultdict of int, the number of times each tag occurs as the final tag in a sentence
    - initial_tags_occurrences: defaultdict of int, the number of times each tag occurs as the initial tag in a sentence
    """

    sentence_number = 0
    tag_set = []
    tags_occurrence = defaultdict(int)
    final_tags_occurrence = defaultdict(int)
    initial_tags_occurrences = defaultdict(int)

    for i, line in enumerate(training_dataset):
        if line.split():
            id, _, tag = line.split()
            if id == "0":
                sentence_number += 1
                initial_tags_occurrences[tag] += 1
            if tag not in tag_set:
                tag_set.append(tag)
            tags_occurrence[tag] += 1
        elif i == len(training_dataset) - 1 and line != '\n':
            _, _, tag = line.split()
            final_tags_occurrence[tag] += 1
        else:
            _, _, tag = training_dataset[i - 1].split()
            final_tags_occurrence[tag] += 1


    return sentence_number, sorted(tag_set, key=lambda x: tag_order[x]), tags_occurrence, final_tags_occurrence, initial_tags_occurrences


def get_transition_probabilities_matrix():
    """
    This function computes the transition probabilities between tags.
    :returns a dictionary containing the transition probabilities P(tag|prev_tag) for all tags in the training dataset.
    """
    transition_matrix = {}

    for tag in tag_set:
        transition_matrix[tag] = [EPSILON for el in tag_set]

    for i, line in enumerate(training_dataset):
        if not line.strip():
            continue
        if i == len(training_dataset) - 1:
            continue

        current_id, _, current_tag = line.split()
        if current_id == '0':
            continue

        prev_line = training_dataset[i - 1]
        if not prev_line.strip():
            continue

        prev_id, _, prev_tag = prev_line.split()
        if prev_id == '0':
            continue

        current_tag_index = tag_set.index(current_tag)
        transition_matrix[prev_tag][current_tag_index] += 1

    for tag in transition_matrix:
        for i in range(len(transition_matrix[tag])):
            transition_matrix[tag][i] = transition_matrix[tag][i] / tags_occurrence[tag]

    return transition_matrix


def get_start_probability():
    """
     Computes the initial probability P(TAG|START) for each tag in the training data.
     The initial probability of a tag is calculated as the number of times the tag appears as the first
     tag in a sentence, divided by the total number of sentences in the training data.

     Returns:
     - start_probability: dictionary, the initial probability of each tag in the training data.
       The key is the tag and the value is the probability.
     """
    start_probability = {tag: initial_tags_occurrences.get(tag, 0) / sentence_number for tag in tag_set}
    return start_probability


def get_end_probability():
    """
    This function calculates the probability of each tag being the final tag in a sentence which is used for predicting
    the end of named entities.
    The final probability of a tag is calculated as the number of times the tag appears as the final
    tag in a sentence, divided by the total number of occurrences of the tag in the training data.
    Returns:
    - end_probability: dictionary, the probability of each tag being the final tag in a sentence.
        The key is the tag and the value is the probability.
    """
    end_probability = {tag: final_tags_occurrence.get(tag, 0) / tags_occurrence[tag] for tag in final_tags_occurrence}
    # Add tags with zero final occurrences
    end_probability.update({tag: EPSILON for tag in tag_set if tag not in end_probability})
    return end_probability


def smoothing_4():
    """
    This function calculates the emission probability for words that appear only once in the validation set.
    Returns:
        - emission_probs: dictionary, where keys are words and values are lists of emission
        probabilities for each tag in tag_set

    """
    # read the validation dataset
    dataset_lines = read_dataset(language + '/val.conllu')

    # initialize word_count and occurrences
    word_count = {}
    occurrences = {}
    # looping over dataset_lines
    for line in dataset_lines:
        if line.split():
            _, word, tag = line.split()
            # count word occurrences
            if word not in word_count:
                word_count[word] = 0
            word_count[word] += 1
            # count tag occurrences for each word, searching for words which appears only once
            if word_count[word] == 1:
                if word not in occurrences:
                    # initialize the tag occurrences to 0 for the new word
                    occurrences[word] = [0 for _ in tag_set]
                if tag in tag_set:
                    # increment the count for the tag in the occurrences list
                    occurrences[word][tag_set.index(tag)] += 1

    # calculate emission probabilities
    emission_probs = {}
    for word, counts in occurrences.items():
        probs = []
        for i, count in enumerate(counts):
            if count == 0:
                probs.append(EPSILON)
            else:
                probs.append(count / sum(counts))
        emission_probs[word] = probs

    return emission_probs


def most_common_list(emission_probs):
    """
    This function returns the most common tag sequence for words that appear only once in the validation set.
    es. {key1: [0 0 1 0 0 0], key2: [1 0 0 0 0 0]} where key iis a word
    Args:
        emission_probs: a dictionary containing the emission probabilities for each word that appears only once
        in the validation set.
    Returns:
        A list of the most common tag sequence for words that appear only once in the validation set.
    """
    # Create a dictionary to store the counts of each key (tuple of emission probabilities)
    counts = {}
    for word, probs in emission_probs.items():
        # Convert the emission probabilities to a tuple to use it as a dictionary key
        key = tuple(probs)
        if key in counts:
            # If the key exists in the dictionary, append the word to the list of words with the same emission
            # probabilities
            counts[key].append(word)
        else:
            # If the key doesn't exist in the dictionary, create a new list with the word
            counts[key] = [word]

    # Get the key with the highest count
    most_common_key = max(counts)
    # Return the list of emission probabilities (the key)
    return list(most_common_key)


def get_emissions_probability(observations, smoothing):
    """
    Compute the emission probabilities for each word in the training dataset based on the given smoothing method.
    Args:
        observations: a list of words to evaluate
        smoothing: a string indicating which smoothing method to use.
        Possible values are "smoothing_1", "smoothing_2", "smoothing_3", and "smoothing_4".
    Returns:
        emissions: a dictionary containing the emission probabilities for each word in the training dataset.
    """

    # Validate the smoothing parameter
    if not smoothing in ["smoothing_1", "smoothing_2", "smoothing_3", "smoothing_4"]:
        raise ValueError(
            "Wrong smoothing parameter.\nUse one of: ['smoothing_1', 'smoothing_2', 'smoothing_3', 'smoothing_4'])")

    emissions = {}

    # compute the emission probabilities for each word in the training dataset
    if len(emissions) == 0:
        for line in training_dataset:
            if line.split():
                _, word, tag = line.split()
                tag_index = tag_set.index(tag)
                if word not in emissions:
                    emissions[word] = [0 for el in tag_set]
                    emissions[word][tag_index] = 1
                else:
                    emissions[word][tag_index] += 1

        for word in emissions:
            for i in range(len(emissions[word])):
                emissions[word][i] = emissions[word][i] / tags_occurrence[tag_set[i]]

    # handle the case where there are unknown words in the input observations
    exclusive = set(observations) - set(emissions.keys())
    if len(exclusive) == 0:
        return emissions
    elif len(exclusive) != 0 and smoothing == "smoothing_4":
        emission_probs_dict = smoothing_4()
    for sent in exclusive:
        # the emission probabilities are set to 1 for the "O" tag and 0 for all other tags
        if smoothing == "smoothing_1":
            emissions[sent] = [1 if tag == "O" else 0 for tag in tag_set]
        # the probability is set to 0.5 for the "O" and "B-MISC" tags and 0 for all other tags
        elif smoothing == "smoothing_2":
            emissions[sent] = [0.5 if tag in ["O", "B-MISC"] else 0 for tag in tag_set]
        # the probability is set to 1/number_of_tags for all tags
        elif smoothing == "smoothing_3":
            emissions[sent] = [1 / len(tag_set) for tag in tag_set]
        # use the most common tag for words which appear only once in the validation set
        elif smoothing == "smoothing_4":
            most_common_prob = most_common_list(emission_probs_dict)
            emissions[sent] = most_common_prob

    return emissions


def viterbi(observations_list, smoothing):
    """
    Perform Viterbi algorithm to find the most probable sequence of tags given a sequence of observations.
    Args:
        observations_list: a list of words to evaluate.
        smoothing: a string indicating which smoothing method to use. Possible values are "smoothing_1",
                         "smoothing_2", "smoothing_3", and "smoothing_4".
    Returns:
        best_path: list containing the most probable sequence of tags corresponding to the given observations.

    """
    # get the emission probabilities based on the given smoothing method
    emissions = get_emissions_probability(observations_list, smoothing)

    # initialize the Viterbi matrix and the backpointer
    num_tags = len(tag_set)
    num_obs = len(observations_list)
    viterbi_matrix = np.zeros((num_tags, num_obs), dtype=float)
    backpointer = np.zeros((num_tags, num_obs), dtype=int)

    # set the initial probabilities in the Viterbi matrix
    for tag_index in range(num_tags):
        viterbi_matrix[tag_index, 0] = np.log(start_probability[tag_set[tag_index]] + EPSILON) + np.log(
            emissions[observations_list[0]][tag_index] + EPSILON)

    # iterate over the observations and update the Viterbi matrix and backpointer
    for obs_index in range(1, num_obs):
        for tag_index in range(num_tags):
            # compute the product of the previous probabilities, tag transitions, and emission probabilities
            previous_probs = viterbi_matrix[:, obs_index - 1]
            tag_transitions = transition_matrix[tag_set[tag_index]]
            obs_emission = emissions[observations_list[obs_index]][tag_index]
            # the probability of each tag sequence, given the observations and the previously calculated probabilities
            product = [previous_probs[i] + np.log(tag_transitions[i] + EPSILON) + np.log(obs_emission + EPSILON) for i
                       in range(len(previous_probs))]

            # choose the maximum product and update the Viterbi matrix and backpointer
            max_index = np.argmax(product)
            viterbi_matrix[tag_index, obs_index] = viterbi_matrix[max_index, obs_index - 1] + np.log(
                transition_matrix[tag_set[tag_index]][max_index] + EPSILON) + np.log(obs_emission + EPSILON)
            backpointer[tag_index, obs_index] = max_index

    # backtrack from the last observation to the first in order to get the most probable sequence of tags
    best_path = ["" for el in observations_list]
    final_probs = [viterbi_matrix[i, num_obs - 1] + np.log(end_probability[tag_set[i]] + EPSILON) for i in
                   range(num_tags)]
    max_index = np.argmax(final_probs)

    for obs_index in range(num_obs - 1, -1, -1):
        best_path[obs_index] = tag_set[max_index]
        max_index = backpointer[max_index, obs_index]

    return best_path


def compute_baseline(observation_list, smoothing_tag):
    """
    This function takes two arguments: an observation_list containing a list of words to tag,
    and a smoothing_tag that is used when a word is not seen in the training dataset.
    Args:
        observation_list: a list of words to tag
        smoothing_tag: used when a word is not seen in the training dataset
    Returns: the predicted sequence of NER tags

    """
    # will be used to store the count of each NER tag for each word in the training dataset
    baseline_occurrences = collections.defaultdict(collections.Counter)

    # iterates through each line in the training_dataset, and for each word
    # increments the corresponding count in baseline_occurrences
    for line in training_dataset:
        if line and line != '\n':
            _, word, tag = line.split()
            baseline_occurrences[word][tag] += 1

    # create a dictionary to keep track of the frequency of each tag and the most frequent tag for each word
    tag_freq = collections.Counter()
    frequent_tags = {}
    for word, tags in baseline_occurrences.items():
        frequent_tags[word] = tags.most_common(1)[0][0]
        tag_freq.update(tags)

    # gets the most common NER tag across all words in the training dataset:
    # tag_freq is a Counter object that contains the frequency of each NER tag across all words in the training dataset
    # most_common(1) method returns a list of tuples where the first element is the most common tag es. ('O', 2066996)
    # and the second element is its frequency, [0][0] is then used to access the most common tag
    # from the first tuple in the list (first tuple, first element which is the tag)
    most_common_tag = tag_freq.most_common(1)[0][0]

    # storing here the predicted NER tags for each word in observation_list
    best_path = []

    for obs in observation_list:
        # check if the word is in the training dataset
        if obs in frequent_tags:
            # use the most frequent tag for that word
            best_path.append(most_common_tag)
        else:
            # if the word is not in the training dataset, use the smoothing tag which is B-MISC
            best_path.append(smoothing_tag)

    # the predicted sequence of NER tags is then returned
    return best_path


def compute_decoding(smoothing, baseline):
    """
    The function, for each sentence,computes the most likely sequence of NER tags using either the Viterbi algorithm or
    a baseline approach. It then obtains the correct NER tag sequence from the parsed test dataset, and uses the
    compute_scores function to compare the predicted and correct sequences and update performance metrics.
    Finally, the function returns the accuracy of the predicted NER tags.
    Args:
        smoothing: a parameter used in the Viterbi function
        baseline: a boolean indicating whether to use the baseline approach
    Returns:
        accuracy: a float indicating the accuracy of the predicted NER tags
    """

    predicted_sequence = []
    correct_sequence = []

    # reading in the test dataset and parsing it into a list of sentences.
    test_dataset = read_dataset(language + '/test_hp_en.txt')  # test_hp_en.txt - test.conllu // CHANGE
    sentences = []
    # each sentence is represented as a list of tuples, where each tuple contains a word and its associated NER tag.
    sentence = []
    for line in test_dataset:
        if line.split():
            _, word, tag = line.split()
            sentence.append((word, tag))
        else:
            sentences.append(sentence)
            sentence = []
    sentences.append(sentence)
    # Determine the number of sentences to evaluate in the compute_decoding function.
    # If number_of_sentences is not provided, it evaluates all sentences in the test dataset
    # otherwise, it evaluates the number of sentences specified in number_of_sentences.
    number_of_sentences = None  # default value
    num_sentences = (len(sentences) + 1) if number_of_sentences is None else number_of_sentences

    # compute the most likely sequence of NER tags for the sentence using the Viterbi algorithm or the baseline approach
    # the correct NER tag sequence for the sentence is obtained from the parsed test dataset, and the `compute_scores`
    # function is called to compare the predicted and correct sequences and update performance metrics.
    for tagged_couples in sentences[:num_sentences]:
        splitted_sentences = [el[0] for el in tagged_couples]
        if baseline == 1:
            best_path = compute_baseline(splitted_sentences, 'B-MISC')
        else:
            best_path = viterbi(splitted_sentences, smoothing)
        correct_path = [couple[1] for couple in tagged_couples]
        predicted_sequence.extend(best_path)
        correct_sequence.extend(correct_path)
        compute_scores(correct_path, best_path)

    # print(predicted_sequence)
    # compute the accuracy of the predicted NER tags by counting the number of correct predictions and dividing
    # by the total number of predictions.
    check_correct = [i for i, j in zip(predicted_sequence, correct_sequence) if i == j]
    accuracy = len(check_correct) / len(predicted_sequence)

    return accuracy


" SCORES "


def compute_scores(correct_path, predicted_path):
    # accuracy of tag O (the only one with no B-x, I-x)
    total_o = len([i for i in range(len(correct_path)) if correct_path[i] == "O"])
    # check how many 'O' correct O predicted
    check = len(
        [i for i in range(len(correct_path)) if (correct_path[i] == predicted_path[i]) and correct_path[i] == "O"])
    accuracy_data["O"]["TP"] += check
    accuracy_data["O"]["TOT"] += total_o

    # precision and recall of tag O
    if accuracy_data["O"]["TP"] == 0:
        accuracy_data["O"]["PRECISION"] = 0
        accuracy_data["O"]["RECALL"] = 0
    else:
        # calculates the precision and recall of the "O" tag.
        # The precision is calculated as the ratio of TP to the sum of TP and FP.
        # The recall is calculated as the ratio of TP to the total number of actual positives.
        accuracy_data["O"]["PRECISION"] = math.trunc(100 * accuracy_data["O"]["TP"] / (accuracy_data["O"]["TP"] + len([i for i in range(len(correct_path)) if predicted_path[i] == "O" and correct_path[i] != "O"])))
        accuracy_data["O"]["RECALL"] = math.trunc(100 * accuracy_data["O"]["TP"] / accuracy_data["O"]["TOT"])

    # accuracy of O
    accuracy_data["O"]["ACCURACY"] = str(math.trunc(100 * (accuracy_data["O"]["TP"] / (accuracy_data["O"]["TOT"])))) + "%"

    # calculating accuracy, precision and recall on all the other entities:
    compute_tags_score(correct_path, predicted_path, "ORG", "B-ORG", "I-ORG")
    compute_tags_score(correct_path, predicted_path, "PER", "B-PER", "I-PER")
    compute_tags_score(correct_path, predicted_path, "MISC", "B-MISC", "I-MISC")
    compute_tags_score(correct_path, predicted_path, "LOC", "B-LOC", "I-LOC")


def compute_tags_score(correct_path, predicted_path, entity_tag, bTag, iTag):
    """
    compute the accuracy score and other evaluation metrics for named entity recognition.
    Args:
        correct_path: a list of correct entity tags for each token in the sentence.
        predicted_path: a list of predicted entity tags for each token in the sentence.
        entity_tag: the type of named entity being evaluated (e.g. "PER", "ORG", "LOC").
        bTag : the tag for the beginning of an entity (e.g. "B-PER", "B-ORG", "B-LOC").
        iTag: the tag for the continuation of an entity (e.g. "I-PER", "I-ORG", "I-LOC").
    Returns:
        score_metrics: a dictionary containing the evaluation metrics (TP, FP, FN, precision, recall).
    """
    # compute accuracy for each entity:
    for i in range(len(correct_path)):
        if (correct_path[i] == bTag) or correct_path[i] == iTag:
            accuracy_data[entity_tag]["TOT"] += 1
        # if prediction is correct --> TP
        if ((correct_path[i] == bTag) and (predicted_path[i] == bTag)) or (
                (correct_path[i] == iTag) and (predicted_path[i] == iTag)):
            accuracy_data[entity_tag]["TP"] += 1

    entities = []  # all the entity in the correct list of tag
    predicted_entities = []  # all the entity tagged by the NER tagger

    s = 0
    while s < len(correct_path):
        # if i-th tag is a B-x
        # es: if B-PER (correct tag for word) == B-PER (considered tag)
        if correct_path[s] == bTag:
            # next position
            j = s + 1
            check = False
            # fin quando j è minore della lunghezza dei tag delle parole considerate
            # e il j-esimo tag (corretto) è uguale a I-x
            while j < len(correct_path) and correct_path[j] == iTag:
                j += 1
                check = True  # start check
            if check:
                entities.append((bTag, s, j - 1))
                s = j
            else:
                # c'è B-PER ma non I-PER
                entities.append((bTag, s, s))
        s += 1

    # predictions
    indx = 0
    # for all the predicted tags
    while indx < len(predicted_path):
        # if I have a B-x tag
        if predicted_path[indx] == bTag:
            # eg. at index 27 I have B-PER, at index 28 I-PER
            j = indx + 1
            check = False
            while j < len(predicted_path) and predicted_path[j] == iTag:
                j += 1
                check = True
            if check:
                # append the tag that starts with B and see where B-PER, I-PER, I-PER ends
                predicted_entities.append((bTag, indx, j - 1))
                # re-move to the last position in order to examine the next tag
                indx = j - 1
            else:
                predicted_entities.append((bTag, indx, indx))
        indx += 1

    # calculates true positives as all those correctly labeled entities
    # calculate false negatives as all those entities not tagged by the NER tagger (tag O)
    for el in entities:
        # if prediction exists in list
        if el in predicted_entities:
            score_metrics[entity_tag]["TP"] += 1
        else:
            score_metrics[entity_tag]["FN"] += 1

    #   calculates false positives as:
    #   tagged entities predicted positive, but actually they are negative
    #   test result that tells you a condition is present, when in reality, there is not

    for el in predicted_entities:
        if not el in entities: score_metrics[entity_tag]["FP"] += 1


def compute_data():
    """
    Computes the precision, recall, and accuracy metrics for each entity tag, and updates
    the corresponding values in the `score_metrics` and `accuracy_data` dictionaries.
    """
    for entity, metric in score_metrics.items():
        # exclude entity tags with "O"
        if entity != "O":
            # calculate precision and recall if there are true positives (TP) or false positives (FP)
            if (metric["TP"] + metric["FP"]) > 0:
                precision_denominator = metric["TP"] + metric["FP"]
                recall_denominator = metric["TP"] + metric["FN"]
                # calculate precision
                if precision_denominator != 0:
                    precision = math.trunc(100 * (metric["TP"] / precision_denominator))
                else:
                    precision = 0
                # calculate recall
                if recall_denominator != 0:
                    recall = math.trunc(100 * (metric["TP"] / recall_denominator))
                else:
                    recall = 0
                # update precision and recall values in the metric dictionary
                metric["PRECISION"] = str(precision) + "%"
                metric["RECALL"] = str(recall) + "%"
            else:
                # If no TP or FP, set precision and recall to 0
                metric["PRECISION"] = "0%"
                metric["RECALL"] = "0%"

            # calculate accuracy if there are total tags (TOT)
            if accuracy_data[entity]["TOT"] > 0:
                accuracy = math.trunc(100 * (accuracy_data[entity]["TP"] / accuracy_data[entity]["TOT"]))
                # update accuracy value in the accuracy_data dictionary
                accuracy_data[entity]["ACCURACY"] = str(accuracy) + "%"
            else: 
                # if no total tags (TOT), set accuracy to 0
                accuracy_data[entity]["ACCURACY"] = "0%"


# accuracy score
accuracy_data = {"ORG": {"TOT": 0, "TP": 0, "ACCURACY": 0},
                 "PER": {"TOT": 0, "TP": 0, "ACCURACY": 0},
                 "MISC": {"TOT": 0, "TP": 0, "ACCURACY": 0},
                 "LOC": {"TOT": 0, "TP": 0, "ACCURACY": 0},
                 "O": {"TOT": 0, "TP": 0, "ACCURACY": 0, "PRECISION": 0, "RECALL": 0}}

score_metrics = {"ORG": {"TP": 0, "FN": 0, "FP": 0, "PRECISION": 0, "RECALL": 0},
                 "PER": {"TP": 0, "FN": 0, "FP": 0, "PRECISION": 0, "RECALL": 0},
                 "MISC": {"TP": 0, "FN": 0, "FP": 0, "PRECISION": 0, "RECALL": 0},
                 "LOC": {"TP": 0, "FN": 0, "FP": 0, "PRECISION": 0, "RECALL": 0}}

#read training dataset
training_dataset = read_dataset(language + '/train.conllu')
print("\nLearning..")
sentence_number, tag_set, tags_occurrence, final_tags_occurrence, initial_tags_occurrences = compute_learning()

transition_matrix = get_transition_probabilities_matrix()
start_probability = get_start_probability()
end_probability = get_end_probability()

print("\nFound tags: ", tag_set)
print("\nDecoding..")

# accuracy = number of correct predictions / total number of predictions
accuracy = compute_decoding('smoothing_4', 0)  # baseline == 1
print('\nAccuracy: ', round((accuracy * 100), 2), "%")

# compute in order to print precision and recall for tags
compute_data()

print("\n- Accuracy of tags: \n")
for tag in accuracy_data:
    print(tag, accuracy_data[tag])

print("\n- Precision and recall per entity:")
for tag in score_metrics:
    print(tag, score_metrics[tag])