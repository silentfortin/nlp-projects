import utility

from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn


# Mapping algorithms - bag of words approach
def get_signature(synset, stop_wo):
    """
    get the signature (set of meaningful words) associated with a WordNet synset.
    the signature is computed by concatenating the synset's definition with its examples,
    preprocessing the resulting text and returning the resulting set of words
    :param synset: the WordNet synset for which to compute the signature
    :param stop_words: the set of stopwords to remove from the signature
    :return: the signature associated with the synset, as a set of words
    """
    # get the synset's definition
    definition = synset.definition()
    # concatenate the examples to the definition
    for example in synset.examples():
        definition = definition + " " + example
    # preprocess the text
    signature = utility.text_preprocessing(definition, stop_wo)
    # return the signature as a set of words
    return set(signature)


def get_overlap(signature, context):
    return len(set(signature) & set(context))


def maximize_overlap(target_word, context):
    """
    determine the most likely sense of a target_word in a given context using the maximize overlap method
    :param target_word: the target word to disambiguate
    :param context: a list of words representing the context in which the target_word appears
    :return: the synset that is the most likely sense of the target_word in the context
    """
    # initialize the maximum overlap score
    max_score = 0
    # initialize the best sense
    best_synset = None
    # loop over the synsets of the target word
    for synset in wn.synsets(target_word):
        # get the signature of the synset
        synset_signature = get_signature(synset, stop_words)
        # compute the overlap score between the signature and the context
        overlap_score = get_overlap(synset_signature, context) + 1
        # if the overlap_score is greater than the maximum score so far
        if overlap_score > max_score:
            max_score = overlap_score
            best_synset = synset
    return best_synset



def get_disambiguation_context_for_frame_name(frame, term):
    """
    get disambiguation context for the name of a frame in FrameNet
    :param frame: the frame for which to get the disambiguation context
    :param term: the considered term for the frame name.
    :return: a dictionary containing the ID of the frame, its term, and the computed context
             as a set of preprocessed words
    """
    # concatenate the term and the definition of the frame
    context_data = term + " " + frame.definition
    # preprocess the context using the given stop words
    disambiguated_context = utility.text_preprocessing(context_data, stop_words)
    # return a dictionary with the ID, term, and context
    return {'id': frame.ID, 'word': term, 'context': disambiguated_context}


def get_disambiguation_context_for_frame_elements(frame_elements, list_of_terms):
    """
    get disambiguation context for the Frame Elements of a FrameNet frame.
    Args:
        frame_elements: a dictionary containing the Frame Elements of a frame, where the
            keys are the names of the FEs and the values are fn.frame.FE objects
        list_of_terms: a list of terms to consider when extracting the contexts for the FEs
    Returns:
        a list of dictionaries, one for each selected FE, containing the ID, name, and context
        as a set of preprocessed words
    """
    frame_elements_context = []

    # get the keys in the frame_elements dict that match the terms in the list_of_terms
    keys = [key for key in frame_elements.keys()]
    selected_keys = [el for el in keys if el.lower() in list_of_terms]

    # loop over the selected keys and extract the context for each FE
    for i in range(len(selected_keys)):
        considered_key = selected_keys[i]
        # concatenate the term and the definition of the FE
        context_data = list_of_terms[i] + " " + frame_elements[considered_key].definition
        # preprocess the context using the given stop words
        disambiguated_context = utility.text_preprocessing(context_data, stop_words)
        # add a dictionary with the ID, name, and context to the list
        frame_elements_context.append(
            {'id': frame_elements[considered_key].ID, 'word': frame_elements[considered_key].name.lower(), 'context': disambiguated_context})

    return frame_elements_context


def get_disambiguation_context_for_lexical_units(lexical_units, term_list):
    """
    given a dictionary of lexical units and a list of terms, this function returns a list of dictionaries containing the
    disambiguation context for each lexical unit whose name (without the part-of-speech tag) is in the list of terms.
    The disambiguation context is composed of the term and the definition of the lexical unit, after preprocessing it
    with the function text_preprocessing from the utility module
    :param lexical_units: a dictionary where the keys are the names of the lexical units and the values are the lexical
    unit objects
    :param term_list: a list of terms (strings) to disambiguate
    :return: a list of dictionaries, where each dictionary contains the following keys:
             - 'id': the ID of the lexical unit
             - 'word': the name of the lexical unit in lowercase
             - 'context': the disambiguation context (a set of preprocessed words)
    """
    lexical_units_context = []
    selected_keys = []
    keys = [key for key in lexical_units.keys()]

    # select the keys whose name (without the POS tag) is in term_list
    for key in keys:
        pos_tags = [".v", ".n", ".a", ".s"]
        word_without_pos = key.lower()
        for pos in pos_tags:
            word_without_pos = word_without_pos.replace(pos, '')
        if word_without_pos in term_list:
            selected_keys.append(key)

    # for each selected key, construct the disambiguation context and add it to the result list
    for key in selected_keys:
        lexical_unit = lexical_units[key]
        name = lexical_unit.name.lower()
        definition = lexical_unit.definition
        context_data = f"{name} {definition}"
        disambiguated_context = utility.text_preprocessing(context_data, stop_words)
        lexical_units_context.append(
            {'id': lexical_unit.ID, 'word': name, 'context': disambiguated_context})
    return lexical_units_context



def get_frame_elements_disambiguation_context(frames_list, terms_list):
    """
    get disambiguation context for all frame elements
    :param frames_list: a list of Frame objects
    :param terms_list: a list of dictionaries containing the terms for the frame name, frame elements, and lexical units
    :return: a list of disambiguation contexts for each frame element
    """
    disambiguated_contexts = []
    for i in range(len(frames_list)):
        # get disambiguation context for the frame name
        frame_name_context = get_disambiguation_context_for_frame_name(frames_list[i],
                                                                       terms_list[i]['frame_name'])
        # get disambiguation context for the frame elements
        frame_elements_context = get_disambiguation_context_for_frame_elements(frames_list[i].FE,
                                                                               terms_list[i]['frame_elements'])
        # get disambiguation context for the lexical units
        lexical_unit_context = get_disambiguation_context_for_lexical_units(frames_list[i].lexUnit,
                                                                            terms_list[i]['lexical_units'])
        # append the disambiguation contexts for this frame element to the list
        disambiguated_contexts.append({'frame_name_context': frame_name_context,
                                       'frame_elements_context': frame_elements_context,
                                       'lexical_unit_context': lexical_unit_context})
    return disambiguated_contexts



def map_frames_to_synsets(frames, terms, mapping_func):
    """
    map each element in a list of farmes to its most likely WordNet synset based
    on the disambiguation context found on FrameNet

    :param frames: a list of frames
    :param terms: a list of terms corresponding to the elements in the frames
    :param mapping_func: a function that maps a term to its synset
    :return: a list of dictionaries with mapped terms
    """
    disambiguated_context = get_frame_elements_disambiguation_context(frames, terms)
    mapped_terms = []
    for i in range(len(disambiguated_context)):
        current_frame = disambiguated_context[i]['frame_name_context']
        current_fes = disambiguated_context[i]['frame_elements_context']
        current_lus = disambiguated_context[i]['lexical_unit_context']
        best_frame = {str(current_frame['id']): mapping_func(current_frame['word'], current_frame['context']).name()}
        best_lus = {}

        # finding the best lexical units
        for lexical_unit in current_lus:
            word = lexical_unit['word']
            if '_' in word:
                word = word.split('_', 1)[1]
            if '-' in word:
                word = word.split('-', 1)[1]
            if '(' in word:
                word = word.split('(', 1)[0].replace(" ", "")
            if '.v' in word:
                word = word.replace(".v", '')
            elif '.n' in word:
                word = word.replace(".n", '')
            elif '.a' in word:
                word = word.replace(".a", '')
            elif '.s' in word:
                word = word.replace(".s", '')
            best_sense = mapping_func(word, lexical_unit['context'])
            name = best_sense.name() if best_sense else ''
            best_lus[str(lexical_unit['id'])] = name

        # finding best frame elements
        best_fes = {}
        for frame_element in current_fes:
            word = frame_element['word']
            if '_' in word:
                word = word.split('_', 1)[1]
            if '-' in word:
                word = word.split('-', 1)[1]
            best_sense = mapping_func(word, frame_element['context'])
            name = best_sense.name() if best_sense else ''
            best_fes[str(frame_element['id'])] = name

        mapped_terms.append({'frame_name': best_frame, 'lexical_units': best_lus, 'frame_elements': best_fes})

    return mapped_terms


def evaluate(manual_annotations, auto_annotations):
    """
    evaluates the score comparing manual annotation and generated ones
    :param manual_annotations: list of manual annotations
    :param auto_annotations: list of automatic annotations
    :return: the computed score
    """

    # initialize counters
    num_correct = 0
    num_total = 0

    # loop through each annotation pair
    for i in range(len(manual_annotations)):
        manual_annotation = manual_annotations[i]
        auto_annotation = auto_annotations[i]

        # extract relevant information from each annotation
        manual_frame = manual_annotation['frame_name']
        manual_lexical_units = manual_annotation['lexical_units']
        manual_frame_elements = manual_annotation['frame_elements']

        auto_frame = auto_annotation['frame_name']
        auto_lexical_units = auto_annotation['lexical_units']
        auto_frame_elements = auto_annotation['frame_elements']

        # add to total count for each element in the manual annotation
        num_total += len(manual_frame) + len(manual_lexical_units) + len(manual_frame_elements)

        # check for correct frame annotation
        for frame_name in manual_frame:
            if manual_frame[frame_name] == auto_frame[frame_name]:
                num_correct += 1

        # check for correct lexical unit annotations
        for lu_id in manual_lexical_units:
            if manual_lexical_units[lu_id] == auto_lexical_units[lu_id]:
                num_correct += 1

        # check for correct frame element annotations
        for fe_id in manual_frame_elements:
            if manual_frame_elements[fe_id] == auto_frame_elements[fe_id]:
                num_correct += 1

    # return the ratio of correctly annotated elements to total elements
    return num_correct / num_total


frame_set = utility.get_frame_set_for_student('cogliandro')

# get stopwords
stop_words = stopwords.words('english')

# get used terms
used_terms = utility.get_used_terms('used_terms.json')

mapping_bow = map_frames_to_synsets(frame_set, used_terms, maximize_overlap)
# todo: ricorda la {} in più
#utility.write_to_file("bag_of_words_output.json", [mapping_bow])

# loads files
manual_annotation = utility.load_from_file("manual_annotation.json")
found_annotation = utility.load_from_file("bag_of_words_output.json")

# evaluates
score = evaluate(manual_annotation, found_annotation)
print('\nAccuracy:\t' + str(round(score * 100, 2)) + "%")

