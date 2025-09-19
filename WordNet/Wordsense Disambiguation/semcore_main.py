import string
import nltk
import random
from nltk import WordNetLemmatizer, word_tokenize
from nltk.corpus import semcor, stopwords
from nltk.corpus import wordnet as wn

nltk.download('semcor')
nltk.download('stopwords')

MAX_SENTENCES = len(semcor.sents())


def semcor_sentences_extractor(semcore_path, number_of_sentences):
    """
    return n sentences taken from an annotated corpus of Semcor
    :param semcore_path:
    :param number_of_sentences:
    :return:
    """
    sentences = semcor._items(semcore_path, "token", True, True, True)[:number_of_sentences]
    sentences_list = semcor.sents(semcore_path)[:number_of_sentences]
    return sentences, sentences_list


def semcor_random_sentences_selector(number_of_sentences):
    """
    return n random sentences taken from a set of 37176 sentences in the semcor corpus
    :param number_of_sentences:
    :return:
    """
    # with random index, starting from 0
    end_index = MAX_SENTENCES - number_of_sentences
    random_index = random.randint(0, end_index)
    last_element = random_index + number_of_sentences
    sentences = semcor._items(None, "token", True, True, True)[random_index:last_element]
    sentences_list = semcor.sents()[random_index:last_element]
    return sentences, sentences_list


def read_nn_lemmas(tagged_sentences):
    """
    reading lemmas associated with nouns (NN tag) from a list of sentences taken from semcor corpus
    :param: tagged_sentences:
    :return: a list of elements containing:
                - index of the sentence from which the lemma was extracted
                - the lemma associated with the noun
                - the index of synset associated with the lemma, taken from the annotated corpus
                - ex: [1, 'atmosphere', 1], [2, 'eye', 1]
    """
    list_of_lemmas = []
    for tagged_sentence in range(len(tagged_sentences)):
        # t = tuple
        for t in tagged_sentences[tagged_sentence]:
            if len(t) == 6 and t[1] == "NN" and t[4] is not None and ";" not in t[4]:
                _, _, lemma, _, index_of_synset, _ = t
                # INDEX, LEMMA & SYNSET
                list_of_lemmas.append([tagged_sentence, lemma, index_of_synset])

    return list_of_lemmas


def bag_of_words(sentence):
    """
    BoW
    :param sentence:
    :return:
    """
    lemmatizer = WordNetLemmatizer()
    sentence_no_punctuation = sentence.translate(str.maketrans('', '', string.punctuation)).lower()
    stop_words = set(stopwords.words("english"))
    words = word_tokenize(sentence_no_punctuation)
    filtered_sentence = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return set(filtered_sentence)


def get_context(sense):
    """
    Lesk Algorithm for sense disambiguation
    :param sense:
    :return:
    """
    signature = []
    # getting examples for the sense
    signature += sense.examples()
    # getting definition of the sense
    signature += [sense.definition()]

    for hyp in sense.hypernyms():
        signature += hyp.examples()
        signature += [hyp.definition()]

    for hyp in sense.hyponyms():
        signature += hyp.examples()
        signature += [hyp.definition()]

    # str.join(iterable)
    # returns a string which is the concatenation of the strings in iterable.
    return bag_of_words(" ".join(signature))


def simplified_lesk(word, sentence):
    max_overlap = 0
    senses = wn.synsets(word)
    if senses:
        best_sense = senses[0]
        context = bag_of_words(sentence)
        for sense in senses:
            signature = get_context(sense)
            overlap = len(context.intersection(signature))

            if overlap > max_overlap:
                max_overlap = overlap
                best_sense = sense
    return best_sense


def word_sense_disambiguation(list_of_lemmas, list_of_sentences):
    """
    carries out the disambiguation of the lemma in the context by using Lesk's algorithm for each pair (lemma, sentence)
    then calculate the total accuracy over the set of sentences
    :param list_of_lemmas:
    :param list_of_sentences:
    :return:
    """
    accuracy = 0

    # check if there's any word to disambiguate
    if len(list_of_lemmas) == 0:
        print("No words to disambiguate: no lemma associated with the nouns in the subset of sentences of "
              "the corpus semcor was found.")
        return 0

    for lemma in list_of_lemmas:
        synsets_of_lemma = wn.synsets(lemma[1])
        index_of_synset = int(lemma[2])

        # handling indexes that do not match the synsets associated with the lemmas
        if (index_of_synset == 0) or ((index_of_synset + 1) > len(synsets_of_lemma)):
            continue

        # Lesk Alg
        best_sense_found = simplified_lesk(lemma[1], " ".join(list_of_sentences[lemma[0]]))

        lemma_synset = synsets_of_lemma[index_of_synset - 1]
        if str(lemma_synset) == str(best_sense_found):
            accuracy += 1

    return round(100 * (accuracy / len(list_of_lemmas)), 2)


def word_sense_disambiguation_test(sentences_number, max_number_lemmas, iterations):
    """
    performing a number of disambiguation tests of nouns extracted from the semcor corpus
    :param sentences_number:
    :param max_number_lemmas:
    :param iterations:
    :return:
    """
    final_accuracy = 0
    for i in range(iterations):
        print("Iteretion #" + str(i + 1) + ":")
        tagged_sentences, sentences = semcor_random_sentences_selector(sentences_number)
        nn_lemmas_list = read_nn_lemmas(tagged_sentences)
        lemmas_to_disambiguate = len(nn_lemmas_list) if max_number_lemmas > len(nn_lemmas_list) else max_number_lemmas

        # getting random lemmas to disambiguate
        if lemmas_to_disambiguate >= 2:
            random_elements = random.randint(2, lemmas_to_disambiguate)
            print("random lemmas #: ", random_elements)
            nn_lemmas_list = random.sample(nn_lemmas_list, k=random_elements)
            accuracy = word_sense_disambiguation(nn_lemmas_list, sentences)
            final_accuracy += accuracy
            print("Accuracy at iteretion " + str(i + 1) + ": " + str(accuracy) + "%\n")
        else:
            print("No words to disambiguate: no lemma associated with the nouns in the subset of sentences of "
                  "the corpus semcor was found.\n")

    return round((final_accuracy / iterations), 2)


accuracy = word_sense_disambiguation_test(50, 10, 10)
print("Overall accuracy: " + str(accuracy) + "%")
