# Hanks Theory: Meaning Construction Analysis

This project implements an algorithmic exploration of Patrick Hanks' theory of meaning construction in natural language. The goal is to computationally analyze how verb-argument structures (valency) and lexical relations contribute to the construction of sentence meaning.

## Project Overview

- Focuses on the semantic interplay between verbs and their arguments (subjects/objects), following Hanks' idea that the meaning of a sentence is shaped by selecting argument fillers for verb valency slots.
- Analyzes how different combinations of argument "fillers," categorized by their semantic types (WordNet synsets), generate various interpretations for the verbs considered.

## Methodology

- **Data collection:** Two corpora (one for each target verb, "paint" and "teach") were retrieved from Sketch Engine, each with at least 1000 sentences prioritized by GDEX (good dictionary examples).
- **Preprocessing:** Sentences are cleaned and tokenized, and parts of speech are tagged using NLTK.
- **Slot extraction:** For each sentence, the code extracts the verb, subject, and object. Argument roles (slots) are mapped to their WordNet synsets.
- **Filler and semantic couple analysis:** The system finds semantic "fillers" for each role, and groups them by general parent synsets shared by subject and object. Semantic couples (pairs of fillers for subject and object) are then counted and analyzed for frequency.
- **Result interpretation:** The most frequent semantic pairs for a given verb (e.g., person, artifact for "paint"; group, cognition for "teach") offer insight into typical use cases and range of interpretation according to Hanks' theory.

## Results

- **paint:** Most frequent semantic couple: (person, artifact) — reflects classic “a person paints an object/surface”.
- **teach:** Most frequent semantic couple: (group, cognition) — reflects “teaching a cognitive concept to a group”.
- Less common semantic pairs highlight outlier usages and broader interpretability.

## Repository Structure

- Core processing scripts for slot extraction, semantic analysis, statistics.
- Sample CSV files, processed result files, and supporting functions.

## Requirements

- Python 3.x
- Libraries: nltk, pandas, wordnet
---

Questions and contributions are welcome!
