# NASARI-based Extractive Text Summarization

This project implements an extractive text summarization algorithm leveraging the NASARI semantic vector resource. The aim is to produce concise summaries from input documents using both lexical and semantic relevance measures.

## Project Overview

- The summarizer takes a text and, with a shallow (non-deep-learning) approach, creates an extractive summary by selecting the most relevant sentences.
- **NASARI vectors** are used to compute semantic similarity among terms and sentences.

## Methodology

Two topic detection methods are implemented and compared:
1. **Title Method**: Identifies key content words in the document's title (after punctuation/stopword removal) and determines which sentences are most relevant to these terms.
2. **Cue Phrases Method**: Uses “bonus words” and “stigma words” to assign scores to paragraphs, selecting those with the highest scores as the most relevant.

General workflow:
- Build a NASARI dictionary.
- Read input text files for summarization.
- Choose either the title method or cue phrases method.
- Identify the most important terms/sentences in each document.
- Create a NASARI-based semantic context for relevance scoring.
- Score and rank sentences using weighted overlap similarity.
- Output the summary after reducing the document to a specified percentage.
- Summaries are evaluated using BLEU and ROUGE metrics.

## Results

- Title method: **BLEU score 55.35**, **ROUGE score 52.94**
- Cue phrases method: **BLEU score 45.39**, **ROUGE score 60.29**
- Both approaches have unique strengths and weaknesses, with the title method offering higher topical precision and cue phrases sometimes capturing more relevant details.

## Repository Structure

- Utility modules for:
    - NASARI dictionary construction
    - File reading/writing
    - Summarization and scoring functions
    - BLEU/ROUGE evaluation
- Scripts for experiment execution and example input/output files

## Requirements

- Python 3.x
- Standard scientific Python stack (numpy, pandas, etc.)
---

Contributions and suggestions are welcomed!
