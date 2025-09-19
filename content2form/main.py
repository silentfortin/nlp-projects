import collections
import csv
import string

from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.wsd import lesk
from prettytable import PrettyTable

csv_data = {}
file_path = 'definizioni.csv'
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# function to read data
def read_csv_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        # skip the first row containing participants
        next(reader)
        csv_data = {}
        for row in reader:
            fields = [field.lower().replace("â€™", "'") for field in row[1:]]
            csv_data[row[0].lower()] = fields
    return csv_data

# preprocessing the csv
def preprocess_csv_data(csv_data):
    translator = str.maketrans('', '', string.punctuation)
    for key, value in csv_data.items():
        # tokenize each sentence
        value[:] = [word_tokenize(sentence) for sentence in value]
        # remove single-word sentences
        value[:] = [sentence[0] for sentence in value if len(sentence) > 1]
        # remove stop words and apply lemmatization
        value[:] = [lemmatizer.lemmatize(word.translate(translator)).lower() for word in value
                    if word.lower() not in stop_words]

    return csv_data

# preprocess sentences
def preprocess_sentence(sentence):
    translator = str.maketrans('', '', string.punctuation)
    tokens = word_tokenize(sentence.lower().translate(translator))
    tokens = [token for token in tokens if token not in stop_words]
    lemmas = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmas


# searching most frequent words
def get_most_frequent_words(csv_data):
    most_frequent_words = {}
    for key, value in csv_data.items():
        word_counts = collections.Counter()
        for sentence in value:
            word_counts.update(sentence.split())
        most_frequent_words[key] = dict(word_counts.most_common())
    return most_frequent_words


def fetch_related_synsets(most_frequent_words):
    # dictionary to store the related synsets for each key
    related_synsets = {}
    for key, word_count in most_frequent_words.items():
        # list to store candidate genus synsets
        candidate_genus = []
        # list to store candidate differentia synsets
        candidate_differentia = []
        for word in word_count.keys():
            for pos in [wn.NOUN, wn.VERB, wn.ADJ]:
                # check if the word has at least one synset associated with it for the current part of speech
                if wn.synsets(word, pos=pos) and most_frequent_words[key][word] > 2:
                    for synset in wn.synsets(word, pos=pos):
                        # check if the synset has a hypernym and is not an 'entity' synset
                        if synset.hypernyms() and synset.pos() != 's':
                            candidate_genus.append(synset)
                            for hypernym in synset.hypernyms():
                                candidate_differentia.extend(hypernym.hyponyms())

        related_synsets[key] = {
            'genus': list(set(candidate_genus)),  # store the unique genus synsets for the key
            'differentia': list(set(candidate_differentia))  # store the unique differentia synsets for the key
        }

    return related_synsets


def find_top_synsets(csv_data, preprocessed_sentences, related_synsets):
    # create a dictionary to store the top 5 synsets for each key
    top_synsets = {}

    # disambiguate the meanings of the keys in the context of the related synsets
    for key in csv_data.keys():
        # get the sentences containing the key
        preprocessed_sentence = preprocessed_sentences[key]

        # create a list to store the disambiguated synsets for each sentence
        disambiguated_synsets_per_sentence = []

        # disambiguate the meanings of the words in each sentence
        synsets = []
        for word in preprocessed_sentence:
            # synset_candidates = genus + differentia
            synset_candidates = related_synsets[key]['genus'] + related_synsets[key]['differentia']
            synset = None
            for candidate in synset_candidates:
                if word in candidate.definition().lower():
                    synset = candidate
                    break
            if synset is None:
                synset = lesk(preprocessed_sentence, word)
                if synset is None:
                    synsets_for_word = wn.synsets(word)  # c'è una variazione col pos 'n'
                    if synsets_for_word:
                        synset = synsets_for_word[0]

            synsets.append(synset)

        disambiguated_synsets_per_sentence.append(synsets)

        # find the most frequent synsets across all sentences containing the key
        synset_counts = collections.Counter()
        for synsets in disambiguated_synsets_per_sentence:
            for synset in synsets:
                if synset is not None:
                    synset_counts[synset] += 1

        top_synsets[key] = [s for s in synset_counts.most_common(5) if s[0] is not None]

    return top_synsets


def print_top_synsets(most_related_synsets):
    # create the table
    table = PrettyTable()

    # set the column names
    table.field_names = list(top_synsets.keys())

    # add the column content
    for i in range(5):
        row = []
        for key in top_synsets.keys():
            if len(top_synsets[key]) > i:
                row.append(str(top_synsets[key][i][0]) + ' (' + str(top_synsets[key][i][1]) + ')')
            else:
                row.append('')
        table.add_row(row)

    # print the table
    print(table)

# read the CSV data from the file
csv_data = read_csv_data(file_path)

# preprocess the CSV data
preprocessed_sentences = preprocess_csv_data(csv_data)

# get the most frequent words in each key
most_frequent_words = get_most_frequent_words(preprocessed_sentences)

# get the related synsets for each key (genus)
related_synsets = fetch_related_synsets(most_frequent_words)

# find the top synsets for each key based on disambiguation
top_synsets = find_top_synsets(csv_data, preprocessed_sentences, related_synsets)

# print the top synsets in a table format
print_top_synsets(top_synsets)
