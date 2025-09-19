# WordNet: Conceptual Similarity and Word Sense Disambiguation

This project implements a suite of algorithms for measuring conceptual similarity and performing word sense disambiguation using the WordNet lexical database.

## Project Overview

- Implements several conceptual similarity metrics based on WordNet, including:
    - **Wu & Palmer**
    - **Shortest Path**
    - **Leacock & Chodorow**
- These metrics are tested on word pairs from the `wordsim353` dataset and evaluated against human similarity judgments using Pearson and Spearman correlations.

- Also includes a Word Sense Disambiguation (WSD) system based on the **Simplified Lesk algorithm**, tested on the SemCor annotated corpus.

## Methodology

- **Wu & Palmer:** Calculates similarity based on the depth of the least common subsumer (LCS) between synsets.
- **Shortest Path:** Computes similarity according to the shortest path (in terms of hypernym links) between synsets in the WordNet hierarchy.
- **Leacock & Chodorow:** Uses the shortest path and depth of the taxonomy, applying a logarithmic scaling.
- **Evaluation:** Measures are compared to gold-standard human similarity ratings, with results reported as Pearson and Spearman correlations.

- **Word Sense Disambiguation:** Implements Lesk's algorithm, which chooses the sense of a word in context by maximizing overlap between definitions/examples and sentence context. Overall accuracy is reported on random samples from SemCor.

## Results

- Calculated similarity scores show moderate correlation with human judgments (`Pearson ≈ 28%`, `Spearman ≈ 32%`).
- The WSD system can reach an **accuracy approaching 54%** on favorable samples from the SemCor corpus, though performance varies depending on the random lemma and sentence selection.

## Repository Structure

- Scripts implementing similarity metrics and evaluation routines.
- Scripts for WSD and evaluation on SemCor.
- `wordsin353.csv`: Human-labeled word pair similarity dataset.
- Example output and supporting files.
  
## Requirements

- Python 3.x
- NLTK with WordNet corpus
- Standard scientific Python stack
---

Feedback and contributions welcome!
