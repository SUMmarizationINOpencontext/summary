import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import defaultdict
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from matplotlib.widgets import Slider, Button
import networkx as nx
import itertools
import requests
import json
import xml.etree.ElementTree as ET
import os
import math


# Function to send the entire text to the ISTEX term extraction service
def send_text_to_istex_service(full_text):
    # Calculate the number of words in the full text
    num_words = len(full_text.split())

    # Define the base URL for the term extraction service
    base_url = "https://terms-extraction.services.istex.fr/v1/teeft/en"

    # Construct the full URL with the number of terms parameter equal to the number of words
    url = f"{base_url}?nb={num_words}"

    # Prepare the payload with the full text
    payload = [{"id": 1, "value": full_text}]

    # Send the POST request to the URL with the payload
    response = requests.post(url, json=payload)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON response
        terms = response.json()
        # Extract the terms list from the response
        extracted_terms = terms[0]['value'] if terms and 'value' in terms[0] else []
        return extracted_terms
    else:
        print("Failed to extract terms. Status code:", response.status_code)
        print("Response:", response.text)
        return []
    
def load_custom_stopwords(file_path):
    # Read the stop words from the file, assuming they are separated by paragraphs
    with open(file_path, 'r') as f:
        stop_words = f.read().splitlines()  # Split lines to get each word
    # Return the set of stop words for fast lookups
    return set(stop_words)


def preprocess_all_blocks(blocks, title, frequency_threshold=3):
    lemmatizer = WordNetLemmatizer()
    stop_words = load_custom_stopwords("stopw1.txt")
    preprocessed_blocks = {}
    sentence_mappings = {}

    # Dictionary to store the lemmatized form and original term mapping
    lemmatized_to_original = {}

    # Global frequency count for terms across all blocks
    global_frequencies = defaultdict(int)

    # Helper function to lemmatize and remove stop words from a term
    def clean_and_lemmatize(term):
        words = term.lower().split()
        lemmatized_words = [lemmatizer.lemmatize(word) for word in words if word.lower() not in stop_words and word]
        return ' '.join(lemmatized_words)

    # Step 1: Combine all blocks into a single text to extract terms
    full_text = ' '.join(blocks.values())
    extracted_terms = send_text_to_istex_service(full_text)

    # Step 2: Process terms to lemmatize and remove stop words
    for term in extracted_terms:
        lemmatized_term = clean_and_lemmatize(term)
        if lemmatized_term:
            if lemmatized_term not in lemmatized_to_original:
                lemmatized_to_original[lemmatized_term] = term
            else:
                # If a new term with the same lemmatized form exists, keep the shorter one
                current_term = lemmatized_to_original[lemmatized_term]
                if len(term) < len(current_term):
                    lemmatized_to_original[lemmatized_term] = term

    # Step 3: Process each block to update terms and recalculate frequencies
    for block_name, block_text in blocks.items():
        sentences = sent_tokenize(block_text)
        word_frequencies = defaultdict(int)
        sentence_to_words = []

        # Step 4: Process each sentence in the block
        for sentence in sentences:
            matched_terms = []
            
            for term in extracted_terms:
                if term.lower() in sentence.lower():
                    lemmatized_term = clean_and_lemmatize(term)
                    if lemmatized_term == "":
                        continue
                    # Use the shorter original term
                    original_term = lemmatized_to_original[lemmatized_term]
                    matched_terms.append(original_term)
                    global_frequencies[original_term] += 1  # Update global frequency count

            # Update term frequencies in the block
            for term in matched_terms:
                word_frequencies[term] += 1

            # Store the sentence and its matched terms
            sentence_to_words.append((sentence, matched_terms))

        # Store preprocessed data for the block
        preprocessed_blocks[block_name] = word_frequencies
        sentence_mappings[block_name] = sentence_to_words

    # Apply the title-based frequency boost
    

    # Step 5: Filter terms based on frequency threshold
    for block_name in preprocessed_blocks:
        # Filter out terms that don't meet the threshold
        preprocessed_blocks[block_name] = {term: freq for term, freq in preprocessed_blocks[block_name].items() if global_frequencies[term] > frequency_threshold}

        # Update the sentence mappings to exclude low-frequency terms
        sentence_mappings[block_name] = [
            (sentence, [term for term in terms if global_frequencies[term] > frequency_threshold])
            for sentence, terms in sentence_mappings[block_name]
        ]

    return preprocessed_blocks, sentence_mappings

# Function to multiply the frequency of a term by 50 if all words in the term are in the title
def boost_term_frequency_if_in_title(preprocessed_blocks, title):
    title_set = set(title)  # Convert title to a set for quick lookups

    for block_name, word_frequencies in preprocessed_blocks.items():
        # Loop through each term in the block
        for term in list(word_frequencies.keys()):  # Use list() to avoid changing dict size during iteration
            term_words = term.split()  # Split the term into words

            # Check if all words in the term are in the title
            if all(word in title_set for word in term_words):
                # Multiply the term's frequency by 50
                word_frequencies[term] *= 50

    return preprocessed_blocks


def parse_txt_file(file_path, sections_ignore = ["abstract"]):
    # Open the file and read the content
    with open(file_path, 'r') as f:
        lines = f.readlines()

    sections = {}
    current_section = None
    title = None

    for line in lines:
        stripped_line = line.strip()
        if not stripped_line:
            continue  # Ignore empty lines
        
        # Split the first word as the section name and the rest as text
        splitline = stripped_line.split(" ", 1)
        if len(splitline) > 1:
            current_section = splitline[0]
            text = splitline[1].strip()

            # Save the title section separately
            if current_section.lower() == 'title':
                title = send_text_to_istex_service(text.lower()) # Extract terms from title and send to ISTEX service
            elif current_section.lower() in sections_ignore:
                continue  # Skip ignored sections
            else:
                sections[current_section] = text
    return sections, title

# Function to calculate contrast based on F-measure and global mean
def calculate_contrast(FF_block, FF_bar):
    contrast = FF_block.div(FF_bar, axis=0).fillna(0)
    return contrast

# Function to compute F-measure for words in all blocks and calculate total frequencies
def calculate_word_measures(preprocessed_blocks):
    total_frequencies = defaultdict(int)
    word_measures = {"f_measure": [], "contrast": []}

    # Step 1: Calculate total frequencies across all blocks
    for block_name, block_frequencies in preprocessed_blocks.items():
        for word, count in block_frequencies.items():
            total_frequencies[word] += count

    # Step 2: Calculate F-measure and contrast for each word in each block
    for block_name, block_frequencies in preprocessed_blocks.items():
        block_f_measure = compute_f_measure(block_frequencies, total_frequencies)
        block_f_measures = pd.Series(block_f_measure)
        FF_bar = block_f_measures.mean()

        contrast = calculate_contrast(block_f_measures, FF_bar)

        for word in block_f_measure.keys():
            word_measures["f_measure"].append((word, block_f_measure[word], block_name))
            word_measures["contrast"].append((word, contrast[word], block_name))

    return word_measures


# Function to compute F-measure for words in a block
def compute_f_measure(block_frequencies, total_frequencies):
    f_measures = {}
    total_sum = sum(block_frequencies.values())
    
    for word, count in block_frequencies.items():
        recall = count / total_sum
        predominance = count / total_frequencies[word]
        f_measure = 2 * (recall * predominance) / (recall + predominance) if recall + predominance > 0 else 0
        f_measures[word] = f_measure
    
    return f_measures


# Function to compute F-measure and contrast for each sentence in the blocks
def compute_measures_for_sentences(sentence_mappings, preprocessed_blocks, total_frequencies):
    measures = {"f_measure": [], "contrast": []}
    
    for block_name, sentence_to_words in sentence_mappings.items():
        block_f_measure = compute_f_measure(preprocessed_blocks[block_name], total_frequencies)
        block_f_measures = pd.Series(block_f_measure)
        FF_bar = block_f_measures.mean()
        block_contrast = calculate_contrast(block_f_measures, FF_bar)

        for sentence, words in sentence_to_words:
            f_measure_score = sum(block_f_measures.get(word, 0) for word in words)
            contrast_score = sum(block_contrast.get(word, 0) for word in words)
            measures["f_measure"].append((sentence, f_measure_score, block_name))
            measures["contrast"].append((sentence, contrast_score, block_name))

    return measures


# Function to generate the summary based on the plateau point
def generate_summary(sentence_mappings, preprocessed_blocks):
    # Compute total word frequencies across all blocks
    total_frequencies = defaultdict(int)

    for block_name, block_frequencies in preprocessed_blocks.items():
        for word, count in block_frequencies.items():
            total_frequencies[word] += count
    
    # Compute F-measure and contrast for sentences
    sentence_measures = compute_measures_for_sentences(sentence_mappings, preprocessed_blocks, total_frequencies)
    
    # Generate both F-measure and contrast-based summaries
    summary_f_measure, plateau_f_measure = get_summary_based_on_plateau(sentence_measures['f_measure'])
    summary_contrast, plateau_contast = get_summary_based_on_plateau(sentence_measures['contrast'])
    plot_sentence_measures(sentence_measures, plateau_f_measure, measure_type="f_measure")
    plot_sentence_measures(sentence_measures, plateau_contast, measure_type="contrast")

    return summary_f_measure, summary_contrast, sentence_measures


# Function to get the summary based on plateau detection
def get_summary_based_on_plateau(measure_scores):
    sorted_measures = sorted(measure_scores, key=lambda x: x[1], reverse=True)
    plateau_point = find_plateau_point(sorted_measures)
    summary_sentences = [sentence for sentence, _, _ in sorted_measures[:plateau_point]]
    
    return ' '.join(summary_sentences), plateau_point


def find_plateau_point(combined_measures, decay_rate=0.9):
    # Extract the scores from the combined measures
    scores = [score for _, score, _ in combined_measures]

    # Compute the first derivative (rate of change in scores)
    first_derivative = np.diff(scores)

    # Initialize tolerance and tolerance
    tolerance = np.max(np.abs(first_derivative))  # Start with maximum change allowed

    # Apply exponentially decaying tolerance for subsequent changes
    for idx, delta in enumerate(first_derivative):
        # Update tolerance with exponential decay
        tolerance *= decay_rate

        # Check if the change exceeds the current tolerance
        if np.abs(delta) > tolerance:
            # Significant drop detected, return this index as cutoff
            return idx + 1

    # If no sharp drop is found, return the full length
    return len(scores)

# Function to dynamically adjust the slider and graph for a specific measure
def interactive_plot_word_graph(word_measures, measure_type="f_measure"):
    fig, ax = plt.subplots(figsize=(12, 12))
    plt.subplots_adjust(bottom=0.25)  # Make space for slider

    # Get initial ranges based on the selected measure type (either F-measure or contrast)
    measure_scores = [score for _, score, _ in word_measures[measure_type]]

    # Function to update the graph based on the slider value
    def update_graph(threshold):
        ax.clear()  # Clear the current graph

        # Create a filtered graph based on the current threshold and measure type
        G = create_filtered_word_graph(word_measures, threshold, use_contrast=(measure_type == "contrast"))

        # Redraw the graph
        pos = nx.spring_layout(G, seed=42)
        word_nodes = [node for node, data in G.nodes(data=True) if data['type'] == 'word']
        block_nodes = [node for node, data in G.nodes(data=True) if data['type'] == 'block']

        # Create consistent colors for blocks
        block_colors = {block: color for block, color in zip(block_nodes, itertools.cycle(['red', 'blue', 'green', 'purple', 'orange']))}

        # Draw nodes for words and blocks
        nx.draw_networkx_nodes(G, pos, nodelist=word_nodes, node_color='lightblue', node_size=100, label='Words', ax=ax)
        nx.draw_networkx_nodes(G, pos, nodelist=block_nodes, node_color=[block_colors[block] for block in block_nodes], node_size=300, label='Blocks', ax=ax)

        # Draw edges and display scores in the middle of the edges
        edge_labels = {}
        for u, v, data in G.edges(data=True):
            score = data['score']
            edge_labels[(u, v)] = f'{score:.2f}'  # Score as label in the middle of the edge

        nx.draw_networkx_edges(G, pos, ax=ax)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, font_color='darkgreen', ax=ax)

        # Add labels for block nodes (block names)
        block_labels = {block: block for block in block_nodes}
        nx.draw_networkx_labels(G, pos, labels=block_labels, font_size=10, font_color='darkblue', font_weight='bold', ax=ax)

        # Add labels for word nodes (terms)
        word_labels = {word: word for word in word_nodes}
        nx.draw_networkx_labels(G, pos, labels=word_labels, font_size=8, font_color='black', ax=ax)

        fig.canvas.draw_idle()  # Redraw the graph

    # Initial plot with the minimum measure score
    threshold = min(measure_scores)
    G = create_filtered_word_graph(word_measures, threshold, use_contrast=(measure_type == "contrast"))
    pos = nx.spring_layout(G, seed=42)

    # Create a slider for the chosen measure's threshold
    ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03], facecolor='lightgray')
    slider = Slider(ax_slider, f'{measure_type.capitalize()} Threshold', min(measure_scores), max(measure_scores), valinit=threshold)

    # Function to update the graph when the slider changes
    def slider_update(val):
        threshold = slider.val
        update_graph(threshold)

    slider.on_changed(slider_update)

    plt.show()


# Function to filter and create a word graph based on F-measure or contrast threshold
def create_filtered_word_graph(word_measures, threshold, use_contrast):
    G = nx.Graph()

    # Create nodes for words and blocks
    word_nodes = set()
    block_nodes = set()
    word_to_blocks = defaultdict(set)

    measure_key = "contrast" if use_contrast else "f_measure"

    # Iterate over word measures, add words and blocks based on the threshold
    for word, score, block_name in word_measures[measure_key]:
        if score >= threshold:  # Only consider words with score above the threshold
            word_nodes.add(word)
            block_nodes.add(block_name)
            word_to_blocks[word].add(block_name)

    for block in block_nodes:
        G.add_node(block, type='block')

    # Add nodes and edges only for words that meet the threshold and connect to blocks
    for word in word_nodes:
        G.add_node(word, type='word')
        for block in word_to_blocks[word]:
            weight = 1 / len(word_to_blocks[word])  # Example weight calculation
            score = next(s for w, s, b in word_measures[measure_key] if w == word and b == block)
            G.add_edge(word, block, weight=weight, block=block, score=score)  # Add score as edge attribute

    return G

# Function to plot sentence scores for either F-measure or contrast
def plot_sentence_measures(sentence_measures, plateau_point, measure_type="f_measure"):
    # Sorting and extracting scores based on the selected measure type (F-measure or contrast)
    combined_measures = sorted(sentence_measures[measure_type], key=lambda x: x[1], reverse=True)

    # Extracting x and y values
    x_values = range(1, len(combined_measures) + 1)
    y_values = [score for _, score, _ in combined_measures]

    # Plot the sentence scores
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_values, marker='o', linestyle='-', color='b')
    plt.axvline(x=plateau_point, color='r', linestyle='--', label='Plateau Point')
    plt.title(f"Combined Sentence {measure_type.capitalize()} Scores with Plateau Point")
    plt.xlabel("Sentence Index")
    plt.ylabel(f"{measure_type.capitalize()} Score")
    plt.legend()
    plt.show()

def process_data_pipeline(file_path):
    # Parse and preprocess the txt file
    blocks, title = parse_txt_file(file_path)
    preprocessed_blocks, sentence_mappings = preprocess_all_blocks(blocks, title)
    preprocessed_blocks_title_boosted = boost_term_frequency_if_in_title(preprocessed_blocks, title)

    # Calculate word F-measures and contrast

    word_measures = calculate_word_measures(preprocessed_blocks_title_boosted)

    # Generate and display both F-measure and contrast-based summaries
    print("With log transformation:")
    summary_f_measure, summary_contrast, sentence_measures = generate_summary(sentence_mappings, preprocessed_blocks_title_boosted)
    print("F-measure based summary:")
    print(summary_f_measure)
    print("\nContrast-based summary:")
    print(summary_contrast)

    # Plot the word graph with interactive metric switching
    interactive_plot_word_graph(word_measures, measure_type="f_measure")
    interactive_plot_word_graph(word_measures, measure_type="contrast")

file_path = os.path.join(os.getcwd(), 'text.txt')
process_data_pipeline(file_path)