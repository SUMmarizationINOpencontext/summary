import os
import json
import re
import nltk
import pandas as pd
import numpy as np
import requests
from collections import defaultdict
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer

# Ensure necessary NLTK data is downloaded
nltk.download('wordnet', quiet=True)
nltk.download('punkt_tab', quiet=True)


def load_custom_stopwords(file_path):
    """
    Load custom stop words from a file.

    Parameters:
        file_path (str): Path to the stop words file.

    Returns:
        set: A set of stop words.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The stop words file {file_path} does not exist.")

    with open(file_path, 'r', encoding='utf-8') as f:
        stop_words = f.read().splitlines()
    return set(stop_words)


def send_text_to_istex_service(full_text, cache_enabled=True, cache_file='istex_terms_cache.json'):
    """
    Send text to the ISTEX term extraction service and retrieve extracted terms.

    Parameters:
        full_text (str): The text to process.
        cache_enabled (bool): Whether to cache results.
        cache_file (str): Path to the cache file.

    Returns:
        list: A list of extracted terms.
    """
    # Calculate the number of words in the full text
    num_words = len(full_text.split())
    base_url = "https://terms-extraction.services.istex.fr/v1/teeft/en"
    url = f"{base_url}?nb={num_words}"
    payload = [{"id": 1, "value": full_text}]

    # Check cache
    if cache_enabled and os.path.exists(cache_file):
        with open(cache_file, 'r', encoding='utf-8') as f:
            cache = json.load(f)
        if full_text in cache:
            return cache[full_text]
    else:
        cache = {}

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        terms = response.json()
        extracted_terms = terms[0]['value'] if terms and 'value' in terms[0] else []
        # Save to cache
        if cache_enabled:
            cache[full_text] = extracted_terms
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache, f)
        return extracted_terms
    except requests.exceptions.RequestException as e:
        print(f"An error occurred during the HTTP request: {e}")
        return []
    except ValueError as e:
        print(f"An error occurred while parsing JSON: {e}")
        return []


def clean_and_lemmatize(term, lemmatizer, stop_words):
    """
    Lemmatize and clean a term by removing stop words.

    Parameters:
        term (str): The term to clean.
        lemmatizer (WordNetLemmatizer): NLTK lemmatizer instance.
        stop_words (set): Set of stop words.

    Returns:
        str: The cleaned and lemmatized term.
    """
    words = term.lower().split()
    lemmatized_words = [
        lemmatizer.lemmatize(word) for word in words
        if word.lower() not in stop_words and word
    ]
    return ' '.join(lemmatized_words)


def preprocess_blocks(blocks, stop_words_file, frequency_threshold=3):
    """
    Preprocess text blocks by extracting and cleaning terms.

    Parameters:
        blocks (dict): Dictionary of text blocks.
        stop_words_file (str): Path to the stop words file.
        frequency_threshold (int): Minimum frequency for terms.

    Returns:
        tuple: Preprocessed blocks and sentence mappings.
    """
    lemmatizer = WordNetLemmatizer()
    stop_words = load_custom_stopwords(stop_words_file)
    preprocessed_blocks = {}
    sentence_mappings = {}
    lemmatized_to_original = {}
    global_frequencies = defaultdict(int)

    # Combine all blocks into a single text for term extraction
    full_text = ' '.join(blocks.values())
    extracted_terms = send_text_to_istex_service(full_text)

    # Process terms to lemmatize and remove stop words
    for term in extracted_terms:
        lemmatized_term = clean_and_lemmatize(term, lemmatizer, stop_words)
        if lemmatized_term:
            # Keep the shortest original term for each lemmatized term
            if lemmatized_term not in lemmatized_to_original or \
               len(term) < len(lemmatized_to_original[lemmatized_term]):
                lemmatized_to_original[lemmatized_term] = term

    # Precompile term patterns for efficient matching
    term_patterns = {
        term: re.compile(r'\b' + re.escape(term.lower()) + r'\b')
        for term in extracted_terms
    }

    # Process each block
    for block_name, block_text in blocks.items():
        sentences = sent_tokenize(block_text)
        word_frequencies = defaultdict(int)
        sentence_to_words = []

        for sentence in sentences:
            matched_terms = []
            sentence_lower = sentence.lower()

            # Efficiently find matched terms
            for term, pattern in term_patterns.items():
                if pattern.search(sentence_lower):
                    lemmatized_term = clean_and_lemmatize(term, lemmatizer, stop_words)
                    if not lemmatized_term:
                        continue
                    original_term = lemmatized_to_original[lemmatized_term]
                    matched_terms.append(original_term)
                    global_frequencies[original_term] += 1

            # Update term frequencies
            for term in matched_terms:
                word_frequencies[term] += 1

            sentence_to_words.append((sentence, matched_terms))

        # Store preprocessed data
        preprocessed_blocks[block_name] = word_frequencies
        sentence_mappings[block_name] = sentence_to_words

    # Filter terms based on frequency threshold
    for block_name in preprocessed_blocks:
        preprocessed_blocks[block_name] = {
            term: freq for term, freq in preprocessed_blocks[block_name].items()
            if global_frequencies[term] > frequency_threshold
        }

        # Update sentence mappings
        sentence_mappings[block_name] = [
            (sentence, [
                term for term in terms if global_frequencies[term] > frequency_threshold
            ])
            for sentence, terms in sentence_mappings[block_name]
        ]

    return preprocessed_blocks, sentence_mappings


def compute_f_measure(block_frequencies, total_frequencies):
    """
    Compute the F-measure for each term in a block.

    Parameters:
        block_frequencies (dict): Term frequencies in the block.
        total_frequencies (dict): Total term frequencies across all blocks.

    Returns:
        dict: F-measure scores for each term.
    """
    f_measures = {}
    total_sum = sum(block_frequencies.values())

    for word, count in block_frequencies.items():
        recall = count / total_sum if total_sum > 0 else 0
        predominance = count / total_frequencies[word] if total_frequencies[word] > 0 else 0
        f_measure = 2 * (recall * predominance) / (recall + predominance) \
            if recall + predominance > 0 else 0
        f_measures[word] = f_measure

    return f_measures


def compute_measures_for_sentences(sentence_mappings, preprocessed_blocks, total_frequencies):
    """
    Compute contrast measures for each sentence.

    Parameters:
        sentence_mappings (dict): Mappings of sentences to terms.
        preprocessed_blocks (dict): Preprocessed blocks with term frequencies.
        total_frequencies (dict): Total term frequencies across all blocks.

    Returns:
        pd.DataFrame: DataFrame containing measures for each sentence.
    """
    measures = []

    for block_name, sentence_to_words in sentence_mappings.items():
        block_f_measure = compute_f_measure(preprocessed_blocks[block_name], total_frequencies)

        for sentence, words in sentence_to_words:
            contrast_score = sum(block_f_measure.get(word, 0) for word in words)
            measures.append({
                'sentence': sentence,
                'contrast': contrast_score,
                'block_name': block_name
            })

    # Convert measures to a Pandas DataFrame
    measures_df = pd.DataFrame(measures)
    return measures_df


def find_plateau_point(combined_measures, decay_rate=0.9):
    """
    Find the plateau point in the measures to determine summary cutoff.

    Parameters:
        combined_measures (list): Sorted list of measures.
        decay_rate (float): Rate at which tolerance decays.

    Returns:
        int: Index of the plateau point.
    """
    scores = [score for score in combined_measures]
    scores_normalized = (scores - np.mean(scores)) / np.std(scores) if np.std(scores) > 0 else scores
    first_derivative = np.diff(scores_normalized)
    tolerance = np.max(np.abs(first_derivative))

    for idx, delta in enumerate(first_derivative):
        tolerance *= decay_rate
        if np.abs(delta) > tolerance:
            return idx + 1

    return len(scores)


def get_summary_based_on_plateau(measures_df, decay_rate=0.9):
    """
    Generate a summary based on plateau detection using contrast.

    Parameters:
        measures_df (pd.DataFrame): DataFrame with sentence measures.
        decay_rate (float): Decay rate for plateau detection.

    Returns:
        tuple: Summary sentences and plateau point index.
    """
    sorted_df = measures_df.sort_values(by='contrast', ascending=False)
    combined_measures = sorted_df['contrast'].tolist()
    plateau_point = find_plateau_point(combined_measures, decay_rate)
    summary_sentences = sorted_df['sentence'].iloc[:plateau_point].tolist()
    return ' '.join(summary_sentences), plateau_point


def generate_summary(sentence_mappings, preprocessed_blocks, decay_rate=0.9):
    """
    Generate summary based on contrast.

    Parameters:
        sentence_mappings (dict): Mappings of sentences to terms.
        preprocessed_blocks (dict): Preprocessed blocks with term frequencies.
        decay_rate (float): Decay rate for plateau detection.

    Returns:
        tuple: Summary, plateau point, and sentence measures DataFrame.
    """
    total_frequencies = defaultdict(int)
    for block_frequencies in preprocessed_blocks.values():
        for word, count in block_frequencies.items():
            total_frequencies[word] += count

    # Compute measures for sentences in a DataFrame
    measures_df = compute_measures_for_sentences(
        sentence_mappings, preprocessed_blocks, total_frequencies
    )

    summary_contrast, plateau_contrast = get_summary_based_on_plateau(
        measures_df, decay_rate=decay_rate
    )

    return summary_contrast


def save_results_to_json(results, output_file='summary_output.json'):
    """
    Save the summary and top words to a JSON file.

    Parameters:
        results (list): List of results with PMID, summary, and top words.
        output_file (str): The file path to save the results.
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)


def process_article(article, stop_words_file, decay_rate=0.9):
    """
    Process a single article and generate summary and top words.

    Parameters:
        article (dict): Dictionary containing article sections.
        stop_words_file (str): Path to the stop words file.
        decay_rate (float): Decay rate for plateau detection.

    Returns:
        dict: Contains PMID, summary, and top 10 words.
    """

    pmid = article.get('pmid', 'Unknown')
    blocks = {
        'introduction': article.get('introduction', ''),
        'methods': article.get('methods', ''),
        'results': article.get('results', ''),
        'discussion': article.get('discussion', ''),
        'conclusion': article.get('conclusion', '')
    }

    # Preprocess and generate summary
    preprocessed_blocks, sentence_mappings = preprocess_blocks(blocks, stop_words_file)
    summary = generate_summary(sentence_mappings, preprocessed_blocks, decay_rate)

    # Calculate top 10 words by contrast
    total_frequencies = defaultdict(int)
    for block_frequencies in preprocessed_blocks.values():
        for word, count in block_frequencies.items():
            total_frequencies[word] += count

    # Aggregate F-measures across all blocks
    word_measures = defaultdict(float)
    for block_name, block_frequencies in preprocessed_blocks.items():
        block_f_measures = compute_f_measure(block_frequencies, total_frequencies)
        for word, f_measure in block_f_measures.items():
            word_measures[word] += f_measure

    # Sort by F-measure and get top 10 words
    top_words = sorted(word_measures.items(), key=lambda x: x[1], reverse=True)[:10]
    top_words = [word for word, _ in top_words]

    return {'pmid': pmid, 'summary': summary, 'top_words': top_words}


def process_dataset(input_json, stop_words_file, output_json='summary_output.json', decay_rate=0.9):
    """
    Process the cleaned dataset from JSON and generate summaries for each article.

    Parameters:
        input_json (str): Path to the input JSON file containing the cleaned dataset.
        stop_words_file (str): Path to the stop words file.
        output_json (str): Path to the output JSON file to save summaries and top words.
        decay_rate (float): Decay rate for plateau detection.
    """
    # Load the preprocessed dataset from the JSON file
    with open(input_json, 'r', encoding='utf-8') as f:
        dataset = json.load(f)  # This will load a dictionary keyed by pmid

    results = []
    for pmid, article in dataset.items():  # Iterate over dictionary items
        # Process each article
        result = process_article(article, stop_words_file, decay_rate)
        results.append(result)

        print(f"Processed Article - PMID: {result['pmid']}")
        print(f"Summary: {result['summary'][:500]}")  # Display first 500 characters of the summary
        print(f"Top 10 Words: {result['top_words']}")

    # Save the results to the output JSON file
    save_results_to_json(results, output_file=output_json)


# Sample Usage
if __name__ == '__main__':
    stop_words_file = 'stopw1.txt'
    input_json = 'processed_articles.json'  # This JSON file should contain the cleaned dataset
    output_json = 'summary_output.json'  # This will store the output summaries and top words
    process_dataset(input_json, stop_words_file, output_json)