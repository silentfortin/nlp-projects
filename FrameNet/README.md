# FrameNet-based Word Sense Disambiguation

This project implements a Word Sense Disambiguation (WSD) system using linguistic resources from FrameNet and WordNet. The goal is to assign the correct sense to ambiguous terms by leveraging structured semantic frames and lexical information.

## Project Overview

- The system focuses on selected target terms, identified based on the developer's surname, and uses FrameNet to retrieve relevant frames (for this project: Expansion, Extradition, Mention, Counterattack, Medicalinstruments).
- A manual annotation file (`manualannotation.json`) is created, marking the correct FrameNet frames and linked WordNet synsets for each term.
- Automatic disambiguation is then performed, and results are compared to the manual annotation.

## Methodology

The disambiguation process consists of four main phases:
1. **Data preparation:** Load FrameNet frames and retrieve target terms from `usedterms.json`.
2. **Context generation:** For each frame, frame element, and lexical unit, generate a disambiguation context using definitions and usage examples from WordNet, combined and preprocessed (with stopword removal) to form a "signature."
3. **Disambiguation & mapping:** Compute overlap between these signatures and the context of the ambiguous term, using functions such as `getsignature`, `getoverlap`, and `maximizeoverlap`. Each term is assigned the WordNet synset with maximal contextual overlap.
4. **Evaluation:** The system's automatic annotations are compared with manual annotations, and accuracy is computed.

## Results

- The system achieved an accuracy of **52.78%** in mapping the target terms to their correct senses (frames/synsets) as evaluated on the manually annotated sample.

## Repository Structure

- `manualannotation.json`: Manual annotations of frames, elements, and synsets for each target term.
- `usedterms.json`: List of target terms disambiguated in the project.
- Code implementing data preparation, context generation, disambiguation, and evaluation functions (see scripts in this folder).
- `Relazioni-Radicioni-Esercizio-2.pdf`: Full project report with methodology, implementation details, and discussion.

## Requirements

- Python 3.x
- Standard NLP libraries (nltk, json, etc.)
---

Questions or contributions are welcome!
