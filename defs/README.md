# Defs: Lexical Overlap Analysis of Concept Definitions

This project investigates lexical similarity among definitions of four core concepts, using lexical overlap as a quantitative metric.

## Project Overview

- The objective is to evaluate the lexical overlap among sets of definitions for four selected concepts, accounting for levels of concreteness and specificity.
- Focused concepts: 
    - Brick (concrete-specific)
    - Person (concrete-generic)
    - Revenge (abstract-specific)
    - Emotion (abstract-generic)

## Methodology

- **Preprocessing**: Reads concept definitions from a CSV file, removing punctuation, singletons, and stopwords.
- **Frequency Analysis**: Identifies the most frequent words for each concept's definitions, considering only words above a frequency threshold.
- **Overlap Scoring**: For each word, calculates a score based on its frequency of occurrence across all sentences for a concept.
    - The score is computed as the ratio of the count of a word over the total number of sentences (definitions) in the dataset.
- **Category Scoring**:
    - Computes the average word score separately for abstract-generic, abstract-specific, concrete-generic, and concrete-specific categories.
    - The average is taken over all words with a non-zero score within each category, providing an approximate measure of overall lexical relevance.

## Results

- Concrete concepts (both generic and specific) display greater lexical similarity and overlap in their definitions compared to abstract concepts.
- This suggests that concrete concepts are typically easier to define and are characterized by a higher degree of terminological consistency across definitions.

## Repository Structure

- Scripts for lexical analysis and scoring.
- Input CSV file: definitions for each concept.
- Output tables: overlap scores per concept category.

## Requirements

- Python 3.x
- Standard scientific Python stack (pandas, etc.)

---

Feedback and contributions are welcome!
