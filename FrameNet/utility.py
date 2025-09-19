import hashlib
import json
import string
from random import randint, seed

from nltk import WordNetLemmatizer, word_tokenize
from nltk.corpus import framenet as fn


def get_frames_ids():
    return [f.ID for f in fn.frames()]


def get_frame_set_for_student(student_surname: str, num_frames: int = 5):
    """
    randomly select a set of FrameNet frames for a given student, based on their surname
    :param student_surname: the student's surname, used to determine the base index for frame selection
    :param num_frames: the number of frames to select
    :return: a list of FrameNet frames
    """
    num_total_frames = len(fn.frames())
    base_idx = (abs(int(hashlib.sha512(student_surname.encode('utf-8')).hexdigest(), 16)) % num_total_frames)

    print('Student Surname:', student_surname)
    frame_ids = get_frames_ids()
    i = 0
    offset = 0
    seed(1)
    frames = []

    while i < num_frames:
        frame_id = frame_ids[(base_idx + offset) % num_total_frames]

        # edit to use proper frames
        if frame_id == 77:
            frame = fn.frame(725)
        elif frame_id == 1440:
            frame = fn.frame(2080)
        else:
            frame = fn.frame(frame_id)
        # end

        frame_name = frame.name
        print(f'ID: {frame_id:4d}  Frame: {frame_name}')
        offset = randint(0, num_total_frames)
        i += 1
        frames.append(frame)

    return frames


def get_used_terms(file_path):
    """
    get used terms for frame name, frame elements, and lexical units from a file
    :param file_path: the file path containing the used terms
    :return: the file content in a list
    """
    used_terms = []
    with open(file_path, 'r') as json_file:
        try:
            frames = json.loads(json_file.read())
        except json.decoder.JSONDecodeError:
            print("\nERROR: file is empty")
            exit()

        for frame in frames:
            name = ""
            elements = []
            units = []
            for id in frame["frame_name"]:
                name = frame["frame_name"][id]
            for id in frame["frame_elements"]:
                elements.append(frame["frame_elements"][id])
            for id in frame["lexical_units"]:
                units.append(frame["lexical_units"][id])
            used_terms.append({
                "frame_name": name,
                "frame_elements": elements,
                "lexical_units": units
            })

    return used_terms


def text_preprocessing(text, stop_words_list):
    """
    preprocesses text by lemmatizing, removing punctuation, and filtering out stopwords
    :param text: the input text to preprocess
    :param stop_words: a list of stopwords to remove from the text
    :return: a list of preprocessed words
    """
    preprocessed_text = []

    # initialize lemmatizer
    lemmatizer = WordNetLemmatizer()

    # remove punctuation and convert to lowercase
    text_no_punctuation = text.translate(str.maketrans('', '', string.punctuation)).lower()

    # tokenize the text into words
    words = word_tokenize(text_no_punctuation)

    # lemmatize and filter out stopwords
    for word in words:
        if word.lower() not in stop_words_list:
            preprocessed_text.append(lemmatizer.lemmatize(word))

    return preprocessed_text


def write_to_file(filename, data):
    """
    write data to a JSON file with specified filename
    :param filename: name of the output file
    :param data: data to be written to file
    """
    with open(filename, "w", encoding='utf-8') as file:
        file.write(json.dumps(data, indent=4))


def load_from_file(filename):
    """
    load data from a JSON file with specified filename
    :param filename: name of the input file
    :return: loaded data
    """
    with open(filename, "r", encoding='utf-8') as file:
        try:
            data = json.loads(file.read())
        except json.decoder.JSONDecodeError:
            print("\nERROR: can't read from file " + filename)
            exit()

        return data
