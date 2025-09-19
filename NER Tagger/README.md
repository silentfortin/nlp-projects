# NER Tagger

This project implements a Named Entity Recognition (NER) tagger for English and Italian, developed as part of a Natural Language Processing (NLP) research exercise. The NER tagger automatically detects named entities in text and classifies them into predefined categories such as person, organization, location, etc.

## Dataset

- Primary data source: **WikiNEuRal** corpus (English and Italian datasets already split into train, test, and validation).
- Manual annotation and experiments on a small set of custom Harry Potter-themed sentences (for both English and Italian).

## Methodology

- The system is based on a statistical HMM (Hidden Markov Model) tagging approach.
- Probabilities for transitions, emissions, start, and end tags are estimated from the training corpus.
- Smoothing techniques are applied to handle unseen words/tags and to avoid zero probabilities.
- The decoding of the most likely sequence of tags is performed using the **Viterbi algorithm**.
- A baseline tagger is also implemented, assigning the most frequent tag for each word in the training data or a fallback tag for unknown words.

## Evaluation

- **Metrics:** Overall accuracy, precision, and recall for each named entity category (PER, ORG, LOC, MISC, O).
- The system achieves:
    - Around **92.7% overall accuracy** on both English and Italian (test set).
    - High precision/recall for the O tag, but much lower for actual entity classes (especially PER/ORG).
- Experiments on custom Harry Potter sentences show some improvement with targeted data enrichment, but entity detection performance is limited by training data domain coverage.

## Usage

- See main scripts and notebooks in this folder for data processing, training, tagging, and evaluation.
- Requirements: Python 3.8+, common scientific Python stack (numpy, pandas), no deep learning frameworks required.

## Key Features

- Supports two languages: English and Italian
- Easy to adapt to other CoNLL-style datasets
- Implements multiple smoothing techniques for emissions
- Includes both HMM/Viterbi and simple baseline evaluation

## References

- WikiNEuRal: [Pitz et al., 2021]
- HMM/Viterbi: Standard reference for sequence labeling in NLP
- Project report (in Italian) included for complete methodology, detailed results, and discussion

---

For details and examples, see the PDF report and individual scripts in this folder.
