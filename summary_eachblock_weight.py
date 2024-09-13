import os
import re
import itertools
import requests
import json
import nltk
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
from collections import defaultdict

# Ensure necessary NLTK data is downloaded
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)


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


def preprocess_all_blocks(blocks, stop_words_file, frequency_threshold=3):
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

    # Combine all blocks into a single text
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


def parse_txt_file(file_path, sections_ignore=None):
    """
    Parse a text file into sections.

    Parameters:
        file_path (str): Path to the text file.
        sections_ignore (list): Sections to ignore.

    Returns:
        dict: Dictionary of sections.
    """
    if sections_ignore is None:
        sections_ignore = ["abstract", "title"]

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    sections = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        stripped_line = line.strip()
        if not stripped_line:
            continue
        splitline = stripped_line.split(" ", 1)
        if len(splitline) > 1:
            current_section = splitline[0]
            text = splitline[1].strip()
            if current_section.lower() in sections_ignore:
                continue
            else:
                sections[current_section] = text

    return sections


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


def calculate_contrast(f_measure_series, global_mean_f_measure):
    """
    Calculate the contrast for each term based on F-measure.

    Parameters:
        f_measure_series (pd.Series): F-measure scores for terms.
        global_mean_f_measure (float): Mean F-measure across all terms.

    Returns:
        pd.Series: Contrast values for each term.
    """
    contrast_series = f_measure_series.div(global_mean_f_measure).fillna(0)
    return contrast_series


def calculate_word_measures(preprocessed_blocks):
    """
    Calculate word measures (summed contrast scores) for all terms across blocks.

    Parameters:
        preprocessed_blocks (dict): Preprocessed blocks with term frequencies.

    Returns:
        dict: Word measures containing summed contrast scores.
    """
    total_frequencies = defaultdict(int)
    word_measures_per_block = defaultdict(dict)  # To store per-block contrast scores
    term_total_contrast = defaultdict(float)     # To store total contrast scores per term

    # Calculate total frequencies
    for block_frequencies in preprocessed_blocks.values():
        for word, count in block_frequencies.items():
            total_frequencies[word] += count

    # Calculate F-measure and contrast per block
    for block_name, block_frequencies in preprocessed_blocks.items():
        block_f_measure = compute_f_measure(block_frequencies, total_frequencies)
        block_f_measures = pd.Series(block_f_measure)
        global_mean_f_measure = block_f_measures.mean() if not block_f_measures.empty else 0
        contrast = calculate_contrast(block_f_measures, global_mean_f_measure)

        # Store per-block contrast scores
        for word in block_f_measure.keys():
            word_contrast = contrast[word]
            word_measures_per_block[block_name][word] = word_contrast
            # Sum contrast scores across blocks
            term_total_contrast[word] += word_contrast

    # Update contrast scores in each block to the total contrast
    word_measures = {"contrast": []}
    for block_name, block_terms in word_measures_per_block.items():
        for word in block_terms.keys():
            total_contrast = term_total_contrast[word]
            word_measures["contrast"].append((word, total_contrast, block_name))

    return word_measures


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
        block_f_measures = pd.Series(block_f_measure)
        global_mean_f_measure = block_f_measures.mean() if not block_f_measures.empty else 0
        block_contrast = calculate_contrast(block_f_measures, global_mean_f_measure)

        for sentence, words in sentence_to_words:
            contrast_score = sum(block_contrast.get(word, 0) for word in words)
            measures.append({
                'sentence': sentence,
                'contrast': contrast_score,
                'block_name': block_name
            })

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

    measures_df = compute_measures_for_sentences(
        sentence_mappings, preprocessed_blocks, total_frequencies
    )

    summary_contrast, plateau_contrast = get_summary_based_on_plateau(
        measures_df, decay_rate=decay_rate
    )

    return summary_contrast, measures_df, plateau_contrast


def plot_sentence_measures(measures_df, plateau_point):
    """
    Plot sentence contrast scores and highlight the plateau point.

    Parameters:
        measures_df (pd.DataFrame): DataFrame with sentence measures.
        plateau_point (int): Index of the plateau point.
    """
    sorted_df = measures_df.sort_values(by='contrast', ascending=False)
    x_values = range(1, len(sorted_df) + 1)
    y_values = sorted_df['contrast'].tolist()

    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_values, marker='o', linestyle='-', color='b')
    plt.axvline(x=plateau_point, color='r', linestyle='--', label='Plateau Point')
    plt.title("Sentence Contrast Scores with Plateau Point")
    plt.xlabel("Sentence Index")
    plt.ylabel("Contrast Score")
    plt.legend()
    plt.show()


def create_filtered_word_graph(word_measures, threshold):
    """
    Create a filtered word graph based on a contrast threshold.

    Parameters:
        word_measures (dict): Word measures containing scores.
        threshold (float): Threshold for filtering.

    Returns:
        networkx.Graph: The filtered word graph.
    """
    G = nx.Graph()
    word_nodes = set()
    block_nodes = set()
    word_to_blocks = defaultdict(set)

    for word, score, block_name in word_measures['contrast']:
        if score >= threshold:
            word_nodes.add(word)
            block_nodes.add(block_name)
            word_to_blocks[word].add(block_name)

    for block in block_nodes:
        G.add_node(block, type='block')

    for word in word_nodes:
        G.add_node(word, type='word')
        for block in word_to_blocks[word]:
            weight = 1 / len(word_to_blocks[word])
            score = next(
                s for w, s, b in word_measures['contrast']
                if w == word and b == block
            )
            G.add_edge(word, block, weight=weight, block=block, score=score)

    return G


def interactive_plot_word_graph(word_measures):
    """
    Plot an interactive word graph with a slider to adjust the contrast threshold.

    Parameters:
        word_measures (dict): Word measures containing scores.
    """
    fig, ax = plt.subplots(figsize=(12, 12))
    plt.subplots_adjust(bottom=0.25)

    measure_scores = [score for _, score, _ in word_measures['contrast']]
    min_score, max_score = min(measure_scores), max(measure_scores)

    def update_graph(threshold):
        ax.clear()
        G = create_filtered_word_graph(word_measures, threshold)
        pos = nx.spring_layout(G, seed=42)
        word_nodes = [n for n, data in G.nodes(data=True) if data['type'] == 'word']
        block_nodes = [n for n, data in G.nodes(data=True) if data['type'] == 'block']
        block_colors = {
            block: color for block, color in zip(
                block_nodes, itertools.cycle(['red', 'blue', 'green', 'purple', 'orange'])
            )
        }

        nx.draw_networkx_nodes(G, pos, nodelist=word_nodes, node_color='lightblue',
                               node_size=100, label='Words', ax=ax)
        nx.draw_networkx_nodes(G, pos, nodelist=block_nodes,
                               node_color=[block_colors[block] for block in block_nodes],
                               node_size=300, label='Blocks', ax=ax)

        edge_labels = {}
        for u, v, data in G.edges(data=True):
            score = data['score']
            edge_labels[(u, v)] = f'{score:.2f}'

        nx.draw_networkx_edges(G, pos, ax=ax)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                                     font_size=8, font_color='darkgreen', ax=ax)

        block_labels = {block: block for block in block_nodes}
        nx.draw_networkx_labels(G, pos, labels=block_labels,
                                font_size=10, font_color='darkblue',
                                font_weight='bold', ax=ax)

        word_labels = {word: word for word in word_nodes}
        nx.draw_networkx_labels(G, pos, labels=word_labels,
                                font_size=8, font_color='black', ax=ax)

        fig.canvas.draw_idle()

    threshold = min_score
    update_graph(threshold)

    ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03], facecolor='lightgray')
    slider = Slider(ax_slider, 'Contrast Threshold',
                    min_score, max_score, valinit=threshold)

    slider.on_changed(update_graph)
    plt.show()


def process_data_pipeline(file_path, stop_words_file, decay_rate=0.9):
    """
    Main data processing pipeline.

    Parameters:
        file_path (str): Path to the text file.
        stop_words_file (str): Path to the stop words file.
        decay_rate (float): Decay rate for plateau detection.
    """
    blocks = parse_txt_file(file_path)
    preprocessed_blocks, sentence_mappings = preprocess_all_blocks(blocks, stop_words_file)
    word_measures = calculate_word_measures(preprocessed_blocks)

    # Generate summary based on contrast
    summary_contrast, measures_df, plateau_contrast = generate_summary(
        sentence_mappings, preprocessed_blocks, decay_rate=decay_rate
    )

    print("Summary based on Contrast:")
    print(summary_contrast)

    # Plot sentence measures (contrast only)
    plot_sentence_measures(measures_df, plateau_contrast)

    # Interactive word graph (contrast only)
    interactive_plot_word_graph(word_measures)


if __name__ == '__main__':
    # Configurable file paths
    current_dir = os.getcwd()
    file_path = os.path.join(current_dir, 'text.txt')
    stop_words_file = os.path.join(current_dir, 'stopw1.txt')

    # Process the data
    process_data_pipeline(file_path, stop_words_file)