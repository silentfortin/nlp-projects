import math
import string
from collections import Counter

from nltk import WordNetLemmatizer, word_tokenize
from nltk.corpus import stopwords

stop_words = set(stopwords.words("english"))


def create_nasari_dict():
    """
    reading a given .txt file, this method creates a dictionary. Its elements are in the form:
    {word: {term:score}}
    {'million': '209.35', 'number': '146.31', .. , }
    :return: a dictionary with mapped values
    """
    nasari_dictionary = {}

    with open("dd-small-nasari-15.txt", encoding='utf8') as nasari_file:
        for line in nasari_file:
            line = line.replace("\n", "")
            splitted_line = line.split(";")

            tmp_dict = {}

            for elem in splitted_line[2:]:
                if '_' in elem:
                    value = elem.split("_")
                    tmp_dict[value[0]] = value[1]

            nasari_dictionary[splitted_line[1].lower()] = tmp_dict

    return nasari_dictionary


def read_words(file_name):
    words = []

    with open(file_name, encoding='utf8') as words_file:
        for line in words_file.readlines():
            row = line.replace("\n", "")
            words.append(row.lower())
    return words


def read_document(file_name):
    """
    reading from .txt file the text that needs to be summarized.
    :return: a list containing the text
    """

    clean_document = []
    file_content = []

    with open(file_name, encoding='utf8') as text_file:
        try:
            file_content = text_file.read()
        except FileNotFoundError:
            print("\nERROR: can't read file: " + file_name)
            exit()

    file_lines = file_content.split('\n')

    if len(file_content) != 0:
        for line in file_lines:
            if line != '' and '#' not in line:  # removes wiki link
                line = line.replace("\n", "")
                clean_document.append(line)
    else:
        print("\nERROR: " + file_name + " is empty.")
        exit()

    return clean_document


def write_summarization(file_name, summarized_text):
    with open(file_name, 'w') as file:
        file.truncate(0)
        for line in summarized_text:
            file.write(line)
            file.write('\n')


def title_method(document):
    """
    method to find the most relevant terms in the document's title.
    :param document:
    :return: a list of title words
    """
    return document[0]


def cue_phrases_method(document):
    """
    method to find the most relevant terms in the document
    :param document:
    :return:
    """
    scores_list = []
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))

    bonus_words = read_words("assets/bonus_words.txt")
    stigma_words = read_words("assets/stigma_words.txt")

    lemmatized_words = []

    for i in range(len(document)):

        document_line = document[i]
        document_line_no_punctuation = document_line.translate(str.maketrans('', '', string.punctuation)).lower()

        words = word_tokenize(document_line_no_punctuation)

        for word in words:
            if word not in stop_words:
                lemmatized_words.append(lemmatizer.lemmatize(word))

        # calculate local score summing for every bonus words match +1 and for every stigma words match -1
        local_score = 0
        for lemm_word in lemmatized_words:
            if lemm_word.lower() in bonus_words:
                local_score += 1
            elif lemm_word in stigma_words:
                local_score -= 1

        scores_list.append((local_score, document_line))

    # sorts paragraph for high score. 
    # make the NASARI vector from the bag of words of the higher scored paragraph
    sorted_paragraph = sorted(scores_list, key=lambda x: x[0], reverse=True)
    relevant_paragraph = sorted_paragraph[0][1]

    return relevant_paragraph


def get_bag_of_words(text, stop_words):
    filtered_text = []
    lemmatizer = WordNetLemmatizer()
    # removing punctuation
    text_no_punctuation = str(text).translate(str.maketrans('', '', string.punctuation)).lower()
    # lemmatization and stopwords
    filtered_text = [lemmatizer.lemmatize(word) for word in word_tokenize(text_no_punctuation) if
                     word not in stop_words]
    return set(filtered_text)


def set_relevance_criteria(criteria, document):
    """
    set a relevance criteria choosing between title_method or cue which
    allow us to detect the most important phrases and information
    :param criteria: title_method or cue (Cue phrases method)
    :param document: text which we are analyzing
    :return:
    """

    if criteria == 'title_method':
        return title_method(document)
    elif criteria == 'cue':
        return cue_phrases_method(document)
    else:
        print("\nERROR: this is not a valid method. Choose: 'title_method' or 'cue'")
        exit()


def create_context(text, nasari_vector, stop_words):
    """
    creating a dictionary of NASARI vectors from most relevant terms in the considered
    document starting from a bag of words
    :param text: relevant terms
    :param nasari_vector:
    :return:
    """

    bag_of_words = get_bag_of_words(text, stop_words)
    nasari_vectors = {}

    # finding all the NASARI vectors that are associated to a word in the bag of words
    for word in bag_of_words:
        if word in nasari_vector:
            nasari_vectors[word] = nasari_vector[word]
    return nasari_vectors


def rank(feature, vector):
    """
    rank is defined as the position of the feature within the vector
    :param feature:
    :param vector:
    :return:
    """
    i = 1
    for word in vector:
        if word == feature:
            return i
        i += 1


def weighted_overlap(v1, v2):
    """
    WO method is used as vector comparison method
    :param v1: vector of document's line(s) context
    :param v2: vector of relevant terms context
    :return:
    """
    overlap = 0
    numerator = 0
    denominator = 0

    # computing the overlap between features
    line_features = set(v1.keys())
    relevant_terms_feature = set(v2.keys())
    common_features = list(line_features.intersection(relevant_terms_feature))

    if len(common_features) > 0:

        for feature in common_features:
            numerator += (1 / (rank(feature, v1) + rank(feature, v2)))

        for i in range(len(common_features)):
            denominator += 1 / (2 * (i + 1))

        overlap = numerator / denominator

    return overlap


def sim(w1_dict, w2_dict):
    """
    the similarity between words w1 and w2 is computed as the similarity of their closest senses
    :param w1_dict: w1 list in line_context
    :param w2_dict: w2 list in relevant_terms_context
    :return:
    """
    overlaps = [math.sqrt(weighted_overlap(w1_dict, w2_dict))]
    return max(overlaps)


def summarization(document, document_size_reduction, nasari_dictionary, relevance_criteria):
    """
    computing summarization to the given document

    :param document: considered file
    :param document_size_reduction: percentage of reduction, can be 10% 20% or 30%
    :param nasari_dictionary: elements of nasari vect read from file
    :param relevance_criteria: title method or cue
    :return:
    """

    score = 0

    lines_score = []

    # select the most relevant terms in the document
    relevant_terms = set_relevance_criteria(relevance_criteria, document)

    # create context with the found relevant terms
    relevant_terms_context = create_context(relevant_terms, nasari_dictionary, stop_words)

    for i in range(1, len(document)):
        document_line = document[i]
        line_context = create_context(document_line, nasari_dictionary, stop_words)

        # finding the overlap computing similarity by using Weighted Overlap
        for w1_dict in line_context.values():
            for w2_dict in relevant_terms_context.values():
                # the similarity between words w1 and w2 is computed as the similarity of their closest senses
                score += sim(w1_dict, w2_dict)

        # assigning a score to each document line
        lines_score.append((i, document_line, score))

    # text size after reduction
    new_text_size = len(lines_score) - int(round((document_size_reduction / 100) * len(lines_score), 0))

    # saving only the lines which corresponds to the new-found size
    sorted_lines = sorted(lines_score, key=lambda x: x[2], reverse=True)
    reduced_paragraphs = sorted_lines[:new_text_size]

    # restoring initial lines order using the i-th indices stored before
    # just in case this is not correct

    original_lines_order = sorted(reduced_paragraphs, key=lambda x: x[0])
    summarized_text = [line[1] for line in original_lines_order]

    # adding the title again
    summarized_text.insert(0, document[0])

    return summarized_text


"""
EVALUATION:
    - BLEU
    - ROUGE
"""


def bleu_evaluation(reference, auto_generated):
    """
    calculates BLEU score for text summarization using unigrams
    :param reference: A list of reference sentences
    :param auto_generated: A list of candidate sentences
    :return: The BLEU score (float)
    """

    # remove punctuation and convert words to lowercase for both reference and candidate sentences
    reference = [[word.lower() for word in sentence.translate(str.maketrans('', '', string.punctuation)).split()] for
                 sentence in reference]
    auto_generated = [[word.lower() for word in sentence.translate(str.maketrans('', '', string.punctuation)).split()]
                      for
                      sentence in auto_generated]

    # create a dictionary to store the counts of unigrams in the reference and candidate sentences
    reference_counts = Counter()
    candidate_counts = Counter()

    # loop through the reference sentences and update the reference counts
    for sentence in reference:
        for word in sentence:
            reference_counts[word] += 1

    # loop through the candidate sentences and update the candidate counts
    for sentence in auto_generated:
        for word in sentence:
            candidate_counts[word] += 1

    # compute the clipped count for each unigram: for each unigram by taking the minimum of the count of the unigram
    # in the candidate and reference sentences.
    clipped_counts = {word: min(candidate_counts[word], reference_counts[word]) for word in candidate_counts.keys()}

    # computes the precision score by dividing the sum of the clipped counts by the sum of the candidate counts.
    precision_score = sum(clipped_counts.values()) / sum(candidate_counts.values())

    # computes the brevity penalty by comparing the length of the reference and candidate sentences.
    reference_length = sum(len(sentence) for sentence in reference)
    auto_generated_length = sum(len(sentence) for sentence in auto_generated)
    brevity_penalty = min(1, auto_generated_length / reference_length)

    # computes the BLEU score by multiplying the brevity penalty and the precision score raised to the power
    # of the natural logarithm of the number of unigrams in the candidate sentences
    bleu_score = brevity_penalty * math.exp(math.log(precision_score))

    return bleu_score


def rouge_evaluation(sentences_ref, sentences_gen):
    """
    calculates ROUGE score between two lists of sentences
    :param sentences_ref: list of reference sentences
    :param sentences_gen: list of generated sentences
    :return: ROUGE score for unigrams
    """
    # remove punctuation from reference sentences and create set of unigrams
    unigrams_ref = set(word.lower() for sentence in sentences_ref for word in
                       sentence.translate(str.maketrans('', '', string.punctuation)).split())

    # remove punctuation from generated sentences and create set of unigrams
    unigrams_gen = set(word.lower() for sentence in sentences_gen for word in
                       sentence.translate(str.maketrans('', '', string.punctuation)).split())

    # count number of unigrams in generated sentences that appear in reference sentences
    num_matches = len(unigrams_ref & unigrams_gen)

    # calculate ROUGE score
    recall = num_matches / len(unigrams_ref)

    return recall
