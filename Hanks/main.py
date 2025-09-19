import csv
import re
import string
import nltk
from collections import OrderedDict
from collections import defaultdict
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from prettytable import PrettyTable

file_path = 'paint.csv'
considered_verb = 'paint'


def read_sketch_engine_csv(file_path):
    sentences = []
    with open(file_path, encoding="utf8") as csv_file:
        reader = csv.reader(csv_file)
        row_count = 0
        for row in reader:
            row_count += 1
            if row_count >= 6:
                sentence_id = row_count - 6  # calcola l'ID della sentence
                sentence = row[1]
                # remove <s> and </s> tags from the sentence
                sentence = re.sub(r'<s>', '', sentence)
                sentence = re.sub(r'</s>', '', sentence)
                # remove punctuation from the sentence
                sentence = sentence.translate(str.maketrans('', '', string.punctuation))
                # add the cleaned sentence and its ID to the list of sentences
                sentences.append((sentence_id, sentence.strip()))
    return sentences


def extract_slots(sentences, verb):
    slots = []
    for sentence in sentences:
        # tokenize the sentence
        tokens = nltk.word_tokenize(sentence[1])
        # part-of-speech (POS) tagging
        pos_tags = nltk.pos_tag(tokens)
        # find the index of the verb in the sentence
        verb_index = -1
        for i, (token, pos) in enumerate(pos_tags):
            if token.lower() == verb.lower() and pos.startswith('V'):
                verb_index = i
                break
        # if the verb is found in the sentence
        if verb_index >= 0:
            # extract the subject and object of the verb
            subject = ''
            object = ''
            for j in range(len(pos_tags)):
                if j < verb_index:
                    # check if the token is a noun, adjective, or pronoun and is not already assigned to the subject
                    if pos_tags[j][1][0].lower() in ['n', 'a', 'r'] and pos_tags[j][0] not in subject:
                        # check if the synset exists in WordNet
                        if wn.synsets(pos_tags[j][0], pos=pos_tags[j][1][0].lower()):
                            subject += pos_tags[j][0] + ' '
                elif j > verb_index:
                    # same but for the object
                    if pos_tags[j][1][0].lower() in ['n', 'a', 'r'] and pos_tags[j][0] not in object:
                        # check if the synset exists in WordNet
                        if wn.synsets(pos_tags[j][0], pos=pos_tags[j][1][0].lower()):
                            object += pos_tags[j][0] + ' '
            # strip any trailing spaces from the subject and object
            subject = subject.strip()
            object = object.strip()
            # check if both subject and object are not empty
            if subject and object:
                # get the synset in WordNet
                subject_synsets = wn.synsets(subject)
                object_synsets = wn.synsets(object)
                if subject_synsets and object_synsets:
                    slots.append((sentence[0], subject_synsets[0], object_synsets[0]))
    return slots


def find_fillers(slots):
    # slots = extract_slots(sentences, verb)
    # create a dictionary to keep track of each slot's fillers
    subject_fillers = defaultdict(set)
    object_fillers = defaultdict(set)
    # initialize the WordNet lemmatizer
    lemmatizer = WordNetLemmatizer()
    for sentence_id, subject_synset, object_synset in slots:
        # finds the common parent synset between subject and object
        common_hypernyms = subject_synset.lowest_common_hypernyms(object_synset)
        # take the common_hypernyms name
        common_hypernym = lemmatizer.lemmatize(common_hypernyms[0].name()) if common_hypernyms else None
        # extracts subject and object fillers
        subject_hyponyms = set(subject_synset.hyponyms())
        object_hyponyms = set(object_synset.hyponyms())
        # for each subject filler, check if it's under the common parent synset
        for hyponym in subject_hyponyms:
            if common_hypernym and hyponym in common_hypernyms[0].hyponyms():
                object_fillers[sentence_id].add(hyponym)
            else:
                subject_fillers[sentence_id].add(hyponym)
        # for each object filler, check if it's under the common parent synset
        for hyponym in object_hyponyms:
            if common_hypernym and hyponym in common_hypernyms[0].hyponyms():
                subject_fillers[sentence_id].add(hyponym)
            else:
                object_fillers[sentence_id].add(hyponym)

    return subject_fillers, object_fillers


def find_semantic_couples(subject_fillers, object_fillers, slots):
    lemmatizer = WordNetLemmatizer()
    # create a dictionary to keep track of the frequency of each semantic couple
    semantic_couples_freq = {}

    # for each subject-object slot pair found, look for corresponding fillers for both slots
    for sentence_id, subject_synset, object_synset in slots:
        if sentence_id in subject_fillers and sentence_id in object_fillers:
            for subject_filler in subject_fillers[sentence_id]:
                for object_filler in object_fillers[sentence_id]:
                    if subject_filler in subject_synset.hyponyms() and object_filler in object_synset.hyponyms():
                        # find the common parent synset between subject and object
                        common_hypernyms = subject_synset.lowest_common_hypernyms(object_synset)
                        # get the name
                        common_hypernym = lemmatizer.lemmatize(common_hypernyms[0].name()) if common_hypernyms else None
                        # find the synset of the subject filler
                        subject_filler_synsets = wn.synsets(subject_filler.name().split('.')[0], pos=subject_filler.pos())
                        if subject_filler_synsets:
                            subject_filler_synset = subject_filler_synsets[0]
                            # find the synset of the object filler
                            object_filler_synsets = wn.synsets(object_filler.name().split('.')[0], pos=object_filler.pos())
                            if object_filler_synsets:
                                object_filler_synset = object_filler_synsets[0]
                                # check if the subject filler and object filler have the same common parent synset
                                if common_hypernym and subject_filler_synset.lowest_common_hypernyms(object_filler_synset) and subject_filler_synset.lowest_common_hypernyms(object_filler_synset)[0].name() == common_hypernym:
                                    # create the semantic couple
                                    semantic_couple = (subject_filler.lexname().split(".")[1], object_filler.lexname().split(".")[1])
                                    # update the frequency count of the semantic couple
                                    if semantic_couple in semantic_couples_freq:
                                        semantic_couples_freq[semantic_couple] += 1
                                    else:
                                        semantic_couples_freq[semantic_couple] = 1

    # sort the semantic couples by frequency
    sorted_semantic_couples = sorted(semantic_couples_freq.items(), key=lambda x: x[1], reverse=True)

    return sorted_semantic_couples


def print_semantic_couples_table(sorted_semantic_couples):
    table = PrettyTable()
    table.field_names = ["Semantic Couple", "Frequency"]
    for couple, freq in sorted_semantic_couples:
        table.add_row([couple, freq])
    print(table)


sentences = read_sketch_engine_csv(file_path)
slots = extract_slots(sentences, considered_verb)
fillers = find_fillers(slots)
# <subject_filler, verb, object_filler>
semantic_couples = find_semantic_couples(fillers[0], fillers[1], slots)
print_semantic_couples_table(semantic_couples)

