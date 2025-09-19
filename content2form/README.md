# content2form: Onomasiological Search and Concept Mapping with WordNet

This project implements an onomasiological analysis task using WordNet to identify target concepts starting from their definitions and related lexical material.

## Project Overview

- The goal is to approach a given target term by analyzing multiple definitions associated with a set of provided concepts.
- By leveraging tokenized, lemmatized, and cleaned definitions, the system extracts the most frequent clue words for each concept to infer the genus (general category) and relevant semantic relations.

## Methodology

- **Preprocessing**: Reads concept definitions from a CSV file and performs tokenization, stopword removal, lemmatization, punctuation removal, and lowercasing.
- **Frequent Word Extraction**: Determines the most frequent words for each concept from the cleaned definitions.
- **WordNet Synset Mapping**:
    - For each frequent word, finds associated WordNet synsets and relevant hyponyms (differentia).
    - Builds a concept-specific dictionary mapping key terms to candidate synsets and hyponyms.
- **Word Sense Disambiguation**:
    - Uses definition overlap and the Lesk algorithm to select the most contextually matching synset for each word.
    - Tallies frequencies of each synset across all definitions and selects the top 5 synsets for each concept.
- **Output**: Provides a concise mapping of key concepts to their most representative and frequent WordNet synsets.

## Results

- For each target word (e.g., "emotion"), main synsets reflect various semantic shades and conceptual relationships, revealing both core meanings and peripheral associations.
- Some discrepancies can emerge due to polysemy or ambiguous definitions, requiring careful interpretation of synset candidates.

## Repository Structure

- Main processing scripts for definition analysis and concept-to-synset mapping.
- Input CSV files containing concept definitions.
- Output files listing the top synsets per concept.

## Requirements

- Python 3.x
- NLTK with WordNet corpus
- Standard scientific Python stack

Feedback and contributions are welcome!
