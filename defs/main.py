# Esercitazione #1 - "Defs" - Misurazione dell'overlap lessicale tra una serie di definizioni per concetti
# generici/specifici e concreti/astratti. brick, person, revenge, emotion
import collections
import csv
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from prettytable import PrettyTable

csv_data = {}
file_path = 'definizioni.csv'
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# function to read data from the CSV file
def read_csv_data(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        # skip the first row containing participants
        next(reader)
        csv_data = {}
        for row in reader:
            csv_data[row[0].lower()] = [field.lower() for field in row[1:] if field != "it's"]
    return csv_data

# preprocess the CSV data
def preprocess_csv_data(csv_data):
    translator = str.maketrans('', '', string.punctuation)
    for key, value in csv_data.items():
        # remove single-word sentences and modify the existing list in place
        # value[:] returns a new list that contains all the elements of the original list
        value[:] = [sentence for sentence in value if len(sentence.split()) > 1]
        # remove stop words using generator expression
        value[:] = (' '.join(lemmatizer.lemmatize(word.translate(translator)) for word in sentence.split()
                             if word.lower() not in stop_words) for sentence in value)
    return csv_data


# searching most frequent words
def get_most_frequent_words(csv_data):
    # dictionary to store the most frequent words
    most_frequent_words = {}
    for key, value in csv_data.items():
        # counter object to count word frequencies
        word_counts = collections.Counter()
        for sentence in value:
            # update word frequencies
            word_counts.update(sentence.split())
        most_frequent_words[key] = dict(word_counts.most_common())
    return most_frequent_words

# counting the occurrences of frequent words in the CSV data
def count_frequent_words(csv_data, most_frequent_words):
    word_counts = {}
    for key, value in csv_data.items():
        word_counts[key] = collections.Counter()
        for sentence in value:
            words = sentence.split()
            for word in words:
                if word in most_frequent_words.get(key, {}) and most_frequent_words[key][word] > 3:
                    word_counts[key][word] += 1
    return word_counts

# calculate word scores based on their frequencies in the CSV data
def calculate_word_scores(csv_data, most_frequent_words):
    word_scores = {}
    for key, value in csv_data.items():
        word_counts = collections.Counter()
        for sentence in value:
            words = sentence.split()
            for word in words:
                if word in most_frequent_words.get(key, {}) and most_frequent_words[key][word] > 3:
                    word_counts[word] += 1
        total_sentences = len(value)
        word_scores[key] = {word: word_counts[word] / total_sentences for word in word_counts}
    return word_scores

# calculate the average category score for a given set of keys
def get_average_category_score(scores, keys):
    total_score = 0
    total_words = 0
    for key in keys:
        word_scores = scores.get(key, {})
        for word, score in word_scores.items():
            if score > 0:
                total_score += score
                total_words += 1
    return round(total_score / total_words, 2) if total_words > 0 else 0


csv_data = read_csv_data(file_path)
preproc_csv_data = preprocess_csv_data(csv_data)
most_frequent_words = get_most_frequent_words(preproc_csv_data)

# how many times a word appears in the considered definitions
word_counts = count_frequent_words(csv_data, most_frequent_words)
scores = calculate_word_scores(csv_data, word_counts)

# calculate average word score for 'emotion' and 'revenge'
abstract_gen_avg_score = get_average_category_score(scores, ['emotion'])
abstract_spec_score = get_average_category_score(scores, ['revenge'])
# calculate average word score for 'person' and 'brick'
concrete_gen_avg_score = get_average_category_score(scores, ['person'])
concrete_spec_avg_score = get_average_category_score(scores, ['brick'])

pt = PrettyTable()
pt.field_names = ["", "Astratto", "Concreto"]

pt.add_row(["Generico", abstract_gen_avg_score, concrete_gen_avg_score]),
pt.add_row(["Specifico", abstract_spec_score, concrete_spec_avg_score])
print(pt)
