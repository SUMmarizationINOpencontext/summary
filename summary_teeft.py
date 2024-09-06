# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import defaultdict
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from matplotlib.widgets import Slider
import networkx as nx
import itertools
import requests
import json
import xml.etree.ElementTree as ET
import os



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


def preprocess_all_blocks(blocks, frequency_threshold=1):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    preprocessed_blocks = {}
    sentence_mappings = {}
    
    # Dictionary to store the lemmatized form and original term mapping
    lemmatized_to_original = {}
    
    # Global frequency count for terms across all blocks
    global_frequencies = defaultdict(int)

    # Helper function to lemmatize and remove stop words from a term
    def clean_and_lemmatize(term):
        words = term.lower().split()
        lemmatized_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
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

# Function to compute F-measure for words in all blocks and calculate total frequencies
def calculate_word_f_measures(preprocessed_blocks):
    total_frequencies = defaultdict(int)
    word_f_measures = []

    # Step 1: Calculate total frequencies across all blocks
    for block_name, block_frequencies in preprocessed_blocks.items():
        for word, count in block_frequencies.items():
            total_frequencies[word] += count

    # Step 2: Calculate F-measure for each word in each block
    for block_name, block_frequencies in preprocessed_blocks.items():
        block_f_measure = compute_f_measure(block_frequencies, total_frequencies)
        
        for word, f_measure in block_f_measure.items():
            word_f_measures.append((word, f_measure, block_name))
    
    return word_f_measures

# Function to calculate the mean F-measure across all blocks
def calculate_FF_bar(FF):
    FF_bar = FF.mean()
    return FF_bar

# Function to compute f_measures scores for each sentence in the blocks
def compute_f_measures_for_sentences(sentence_mappings, preprocessed_blocks, total_frequencies, FF_bar):
    f_measures = []
    
    for block_name, sentence_to_words in sentence_mappings.items():
        block_f_measure = compute_f_measure(preprocessed_blocks[block_name], total_frequencies)
        # Filter out words with F-measure lower than FF_bar
        block_f_measure_filtered = {word: f_measure for word, f_measure in block_f_measure.items() if f_measure >= FF_bar[word]}
        block_ff = pd.Series(block_f_measure_filtered)
        
        for sentence, words in sentence_to_words:
            sentence_score = sum(block_ff.get(word, 0) for word in words)  # Now using F-metric directly
            f_measures.append((sentence, sentence_score, block_name))

    return f_measures

# Function to find the plateau point in combined f_measures scores
def find_plateau_point_with_tolerance(combined_f_measures, window_size=5, tolerance=0.1):
    sorted_f_measures = sorted(combined_f_measures, key=lambda x: x[1], reverse=True)
    scores = [score for _, score, _ in sorted_f_measures]
    differences = np.diff(scores)

    # Iterate through the differences with a sliding window
    for i in range(len(differences) - window_size):
        window = differences[i:i + window_size]
        if np.std(window) < tolerance:
            return i + 2  # +2 because index starts from 0, and we want the last stable point

    # If no plateau is found, return the full length
    return len(scores)

# Function to generate the summary based on the plateau point
def generate_summary(sentence_mappings, preprocessed_blocks):
    # Compute total word frequencies across all blocks
    total_frequencies = defaultdict(int)

    for block_name, block_frequencies in preprocessed_blocks.items():
        for word, count in block_frequencies.items():
            total_frequencies[word] += count
    
    # Compute F-measure for each block
    block_f_measures = []
    for block_frequencies in preprocessed_blocks.values():
        block_f_measure = compute_f_measure(block_frequencies, total_frequencies)
        block_f_measures.append(pd.Series(block_f_measure))
    
    FF = pd.DataFrame(block_f_measures).fillna(0)
    FF_bar = calculate_FF_bar(FF)
    
    # Compute f-metric for sentences
    f_measures = compute_f_measures_for_sentences(sentence_mappings, preprocessed_blocks, total_frequencies, FF_bar)
    # Combine all blocks' f_measures scores for global plateau detection
    plateau_point = find_plateau_point_with_tolerance(f_measures, window_size=3, tolerance=0.01)
    
    # Select sentences before the plateau point
    sorted_f_measures = sorted(f_measures, key=lambda x: x[1], reverse=True)
    summary_sentences = [sentence for sentence, _, _ in sorted_f_measures[:plateau_point]]
    
    # Print the sentences in the original order based on the sentence_mappings
    final_summary = []
    for block_name, sentence_to_words in sentence_mappings.items():
        for sentence, words in sentence_to_words:
            if sentence in summary_sentences:
                final_summary.append(sentence)
    
    summary_f_measures = rank_sentences_by_f_measures(f_measures, summary_sentences)
    
    return ' '.join(final_summary), summary_f_measures, plateau_point, f_measures

# Function to rank sentences and keep order by original text
def rank_sentences_by_f_measures(f_measures, summary_sentences):
    # Filter f_measures to include only sentences in the summary
    summary_f_measures = [(sentence, score, block) for sentence, score, block in f_measures if sentence in summary_sentences]
    return summary_f_measures

# Function to filter and create a word graph based on F-measure threshold
def create_filtered_word_graph(word_f_measures, f_threshold):
    G = nx.Graph()
    
    # Create nodes for words and blocks
    word_nodes = set()
    block_nodes = set()
    word_to_blocks = defaultdict(set)
    
    # Iterate over word F-measures, add words and blocks based on the F-measure threshold
    for word, f_score, block_name in word_f_measures:
        if f_score >= f_threshold:  # Only consider words with F-measure above the threshold
            word_nodes.add(word)
            block_nodes.add(block_name)
            word_to_blocks[word].add(block_name)
    
    for block in block_nodes:
        G.add_node(block, type='block')
    
    # Add nodes and edges only for words that meet the F-measure threshold and connect to blocks
    for word in word_nodes:
        G.add_node(word, type='word')
        for block in word_to_blocks[word]:
            weight = 1 / len(word_to_blocks[word])  # Example weight calculation, can be adjusted
            f_measure = next(f for w, f, b in word_f_measures if w == word and b == block)  # Get F-measure for this word-block pair
            G.add_edge(word, block, weight=weight, block=block, f_measure=f_measure)  # Add F-measure as edge attribute
    
    return G

# Function to plot the word graph and integrate a slider for filtering by F-measure
def interactive_plot_word_graph(word_f_measures):
    fig, ax = plt.subplots(figsize=(12, 12))
    plt.subplots_adjust(bottom=0.25)  # Make space for slider

    # Initial plot with no filtering (using minimum F-measure)
    f_threshold = min([score for _, score, _ in word_f_measures])
    G = create_filtered_word_graph(word_f_measures, f_threshold)

    pos = nx.spring_layout(G, seed=42)
    word_nodes = [node for node, data in G.nodes(data=True) if data['type'] == 'word']
    block_nodes = [node for node, data in G.nodes(data=True) if data['type'] == 'block']

    # Create consistent colors for blocks
    block_colors = {block: color for block, color in zip(block_nodes, itertools.cycle(['red', 'blue', 'green', 'purple', 'orange']))}

    # Draw nodes for words and blocks
    nx.draw_networkx_nodes(G, pos, nodelist=word_nodes, node_color='lightblue', node_size=100, label='Words', ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=block_nodes, node_color=[block_colors[block] for block in block_nodes], node_size=300, label='Blocks', ax=ax)
    
    # Draw edges and display F-measures in the middle of the edges
    edge_labels = {}
    for u, v, data in G.edges(data=True):
        f_measure = data['f_measure']
        edge_labels[(u, v)] = f'{f_measure:.2f}'  # F-measure as label in the middle of the edge
    
    nx.draw_networkx_edges(G, pos, ax=ax)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, font_color='darkgreen', ax=ax)

    # Add labels for block nodes (block names)
    block_labels = {block: block for block in block_nodes}
    nx.draw_networkx_labels(G, pos, labels=block_labels, font_size=10, font_color='darkblue', font_weight='bold', ax=ax)

    # Add labels for word nodes (terms)
    word_labels = {word: word for word in word_nodes}
    nx.draw_networkx_labels(G, pos, labels=word_labels, font_size=8, font_color='black', ax=ax)

    # Create a slider for F-measure threshold
    ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03], facecolor='lightgray')
    slider = Slider(ax_slider, 'F-measure', min([score for _, score, _ in word_f_measures]), max([score for _, score, _ in word_f_measures]), valinit=f_threshold)

    # Function to update the graph based on slider value
    def update(val):
        f_threshold = slider.val
        ax.clear()  # Clear current graph
        
        # Recreate and plot graph based on the new F-measure threshold
        G = create_filtered_word_graph(word_f_measures, f_threshold)
        pos = nx.spring_layout(G, seed=42)
        word_nodes = [node for node, data in G.nodes(data=True) if data['type'] == 'word']
        block_nodes = [node for node, data in G.nodes(data=True) if data['type'] == 'block']

        # Draw nodes with consistent colors for blocks
        nx.draw_networkx_nodes(G, pos, nodelist=word_nodes, node_color='lightblue', node_size=100, label='Words', ax=ax)
        nx.draw_networkx_nodes(G, pos, nodelist=block_nodes, node_color=[block_colors[block] for block in block_nodes], node_size=300, label='Blocks', ax=ax)
        
        # Draw edges and display F-measures in the middle of the edges
        edge_labels = {}
        for u, v, data in G.edges(data=True):
            f_measure = data['f_measure']
            edge_labels[(u, v)] = f'{f_measure:.2f}'  # F-measure as label in the middle of the edge
        
        nx.draw_networkx_edges(G, pos, ax=ax)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, font_color='darkgreen', ax=ax)

        # Add labels for block nodes (block names)
        block_labels = {block: block for block in block_nodes}
        nx.draw_networkx_labels(G, pos, labels=block_labels, font_size=10, font_color='darkblue', font_weight='bold', ax=ax)

        # Add labels for word nodes (terms)
        word_labels = {word: word for word in word_nodes}
        nx.draw_networkx_labels(G, pos, labels=word_labels, font_size=8, font_color='black', ax=ax)

        fig.canvas.draw_idle()  # Redraw the graph

    # Update graph when slider is moved
    slider.on_changed(update)
    plt.show()

def plot_sentence_f_measures(blocks, f_measures, plateau_point):
    combined_f_measures = sorted(f_measures, key=lambda x: x[1], reverse=True)
    # Plotting the combined f_measures scores
    x_values = range(1, len(combined_f_measures) + 1)
    y_values = [score for _, score, _ in combined_f_measures]
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_values, marker='o', linestyle='-', color='b')
    plt.axvline(x=plateau_point, color='r', linestyle='--', label='Plateau Point')
    plt.title("Combined Sentence f_measures Scores with Plateau Point")
    plt.xlabel("Sentence Index")


def parse_tei_file(file_path):
    # Parse the TEI XML file
    tree = ET.parse(file_path)
    root = tree.getroot()

    # Namespace dictionary for TEI and other namespaces in the file
    namespaces = {
        'tei': 'http://www.tei-c.org/ns/1.0'
    }

    # Initialize a dictionary to store blocks of text
    blocks = {}

    """# Extract the <abstract> section if it exists
    abstract = root.find('.//tei:abstract', namespaces)
    if abstract is not None:
        # Collect all text in the <abstract>
        abstract_text = ' '.join(''.join(p.itertext()).strip() for p in abstract.findall('tei:p', namespaces))
        blocks['Abstract'] = abstract_text"""

    # Extract all <div> elements in the <body> section of the TEI file
    for div in root.findall('.//tei:body//tei:div', namespaces):
        # Find the first <head> element in the current <div> to use as the block name
        head = div.find('tei:head', namespaces)
        if head is not None and head.text is not None:
            block_name = head.text.strip()
        else:
            continue
        
        # Initialize an empty list to accumulate text content for this block
        block_text = []

        # Loop through all children of the <div> element to extract relevant text
        for elem in div.iter():
            # Collect text from all relevant elements
            if elem.tag in [
                '{http://www.tei-c.org/ns/1.0}p',
                '{http://www.tei-c.org/ns/1.0}ref',
                '{http://www.tei-c.org/ns/1.0}figure',
                '{http://www.tei-c.org/ns/1.0}figDesc',
                '{http://www.tei-c.org/ns/1.0}hi'
            ]:
                # Concatenate all text within the element, including its children
                element_text = ''.join(part.strip() for part in elem.itertext())
                block_text.append(element_text)

        # Join all text components into a single block of text
        blocks[block_name] = ' '.join(block_text)

    return blocks

def remove_similar_sections(blocks):
    # Convert each block of text to a set of unique words
    block_word_sets = {block_name: set(text.lower().split()) for block_name, text in blocks.items()}
    
    # Create a set to store the names of blocks to delete
    blocks_to_delete = set()
    
    # Compare each pair of blocks
    block_names = list(block_word_sets.keys())
    for i in range(len(block_names)):
        for j in range(len(block_names)):
            if i != j:  # Ensure we are not comparing the same block
                name1, name2 = block_names[i], block_names[j]
                set1, set2 = block_word_sets[name1], block_word_sets[name2]
                
                # Ensure we're comparing the smaller set with the larger one
                if len(set1) > len(set2):
                    continue
                
                # Calculate the number of common words
                common_words = len(set1.intersection(set2))
                
                # Check if most words in the smaller set are contained in the larger set
                if common_words >= 0.6 * len(set1):
                    # Mark the smaller block for deletion
                    blocks_to_delete.add(name1)

    # Remove the smaller blocks
    for block_name in blocks_to_delete:
        del blocks[block_name]

    return blocks

# Example usage
file_path = os.path.join(os.getcwd(), 'text.tei')

# Remove similar sections where one has more than half of its words in the other
blocks = remove_similar_sections(parse_tei_file(file_path))

# Preprocess all blocks
preprocessed_blocks, sentence_mappings = preprocess_all_blocks(blocks)

# Generate summary and get the f_measures using preprocessed data
summary, summary_f_measures, plateau_point, f_measures = generate_summary(sentence_mappings, preprocessed_blocks)

print("Generated Summary:")
print(summary)

print("\nNumber of sentences selected based on the combined plateau detection:")
print(plateau_point)

plot_sentence_f_measures(sentence_mappings, f_measures, plateau_point)

interactive_plot_word_graph(calculate_word_f_measures(preprocessed_blocks))