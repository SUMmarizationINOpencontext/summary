# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import defaultdict
from nltk.tokenize import sent_tokenize
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


# Function to preprocess all blocks using the ISTEX service for the entire text
def preprocess_all_blocks(blocks, frequency_threshold=2):
    preprocessed_blocks = {}
    sentence_mappings = {}

    # Combine all blocks into a single full text
    full_text = ' '.join(blocks.values())

    # Send the entire text to ISTEX and get extracted terms
    extracted_terms = send_text_to_istex_service(full_text)
    
    # Initialize a dictionary to count global frequencies across all blocks
    global_frequencies = defaultdict(int)

    # For each block, tokenize into sentences and assign terms
    for block_name, block_text in blocks.items():
        # Tokenize block into sentences
        sentences = sent_tokenize(block_text)
        
        # Initialize data structures for this block
        word_frequencies = defaultdict(int)
        sentence_to_words = []

        # Match extracted terms to sentences within this block
        for sentence in sentences:
            matched_terms = []

            # Check if the term appears in the sentence
            for term in extracted_terms:
                if term.lower() in sentence.lower():
                    matched_terms.append(term)
                    global_frequencies[term] += 1  # Update global frequency count

            # Update word frequencies for the current block
            for term in matched_terms:
                word_frequencies[term] += 1

            # Store the final matched terms for this sentence
            sentence_to_words.append((sentence, matched_terms))
        
        # Store results for this block before filtering
        preprocessed_blocks[block_name] = word_frequencies
        sentence_mappings[block_name] = sentence_to_words

    # Filter terms based on the frequency threshold
    for block_name in preprocessed_blocks:
        # Filter out terms that do not meet the frequency threshold
        preprocessed_blocks[block_name] = {term: freq for term, freq in preprocessed_blocks[block_name].items() if global_frequencies[term] > frequency_threshold}

        # Update sentence_to_words to exclude low-frequency terms
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

# Function to calculate the mean F-measure across all blocks
def calculate_FF_bar(FF):
    FF_bar = FF.mean()
    return FF_bar

# Function to calculate contrast based on F-measure and global mean
def calculate_contrast(FF_block, FF_bar):
    contrast = FF_block.div(FF_bar, axis=0).fillna(0)
    return contrast

# Function to compute contrast scores for each sentence in the blocks
def compute_contrast_for_sentences(sentence_mappings, preprocessed_blocks, total_frequencies, FF_bar):
    contrasts = []
    
    for block_name, sentence_to_words in sentence_mappings.items():
        block_f_measure = compute_f_measure(preprocessed_blocks[block_name], total_frequencies)
        # Filter out words with F-measure lower than FF_bar
        block_f_measure_filtered = {word: f_measure for word, f_measure in block_f_measure.items() if f_measure >= FF_bar[word]}
        block_ff = pd.Series(block_f_measure_filtered)
        
        block_contrast = calculate_contrast(block_ff, FF_bar)
        
        for sentence, words in sentence_to_words:
            # Use the preprocessed words directly to calculate the sentence score
            sentence_score = sum(block_contrast.get(word, 0) for word in words)
            contrasts.append((sentence, sentence_score, block_name))
    return contrasts

# Function to find the plateau point in combined contrast scores
def find_plateau_point_with_tolerance(combined_contrasts, window_size=5, tolerance=0.1):
    sorted_contrasts = sorted(combined_contrasts, key=lambda x: x[1], reverse=True)
    scores = [score for _, score, _ in sorted_contrasts]
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
    
    # Compute contrast for sentences
    contrasts = compute_contrast_for_sentences(sentence_mappings, preprocessed_blocks, total_frequencies, FF_bar)
    # Combine all blocks' contrast scores for global plateau detection
    plateau_point = 4#find_plateau_point_with_tolerance(contrasts, window_size=4, tolerance=0.3)
    
    # Select sentences before the plateau point
    sorted_contrasts = sorted(contrasts, key=lambda x: x[1], reverse=True)
    summary_sentences = [sentence for sentence, _, _ in sorted_contrasts[:plateau_point]]
    
    # Print the sentences in the original order based on the sentence_mappings
    final_summary = []
    for block_name, sentence_to_words in sentence_mappings.items():
        for sentence, words in sentence_to_words:
            if sentence in summary_sentences:
                final_summary.append(sentence)
    
    summary_contrasts = rank_sentences_by_contrast(contrasts, summary_sentences)
    
    return ' '.join(final_summary), summary_contrasts, plateau_point, contrasts

# Function to rank sentences and keep order by original text
def rank_sentences_by_contrast(contrasts, summary_sentences):
    # Filter contrasts to include only sentences in the summary
    summary_contrasts = [(sentence, score, block) for sentence, score, block in contrasts if sentence in summary_sentences]
    return summary_contrasts

# Function to create a word graph for the summary sentences
def create_word_graph(sentence_mappings, summary_contrasts):
    G = nx.Graph()
    
    # Create nodes for words and blocks
    word_nodes = set()
    block_nodes = set(sentence_mappings.keys())
    
    word_to_blocks = defaultdict(set)
    
    for sentence, contrast, block_name in summary_contrasts:
        # Find the corresponding preprocessed words for this sentence
        preprocessed_words = None
        for sent, words in sentence_mappings[block_name]:
            if sent == sentence:
                preprocessed_words = words
                break

        if preprocessed_words is None:
            continue  # If no matching preprocessed words found, skip to the next sentence
        
        block_nodes.add(block_name)
        
        for word in preprocessed_words:
            word_nodes.add(word)
            word_to_blocks[word].add(block_name)
    
    for block in block_nodes:
        G.add_node(block, type='block')
    
    # Add nodes and edges only for words that are in the summary and connected to blocks
    for word in word_nodes:
        G.add_node(word, type='word')
        for block in word_to_blocks[word]:
            # Add an edge between the word and its associated blocks with the 'block' attribute
            weight = 1 / len(word_to_blocks[word])  # Example weight calculation, can be adjusted
            G.add_edge(word, block, weight=weight, block=block)  # Set the 'block' attribute here
    
    return G

def plot_word_graph(G):
    # Extract block nodes
    block_nodes = [node for node, data in G.nodes(data=True) if data['type'] == 'block']
    num_blocks = len(block_nodes)

    # Calculate positions for block nodes in a circular layout
    angle_step = 2 * np.pi / num_blocks
    block_pos = {
        block: (np.cos(i * angle_step), np.sin(i * angle_step))
        for i, block in enumerate(block_nodes)
    }

    # Generate initial positions for the graph with block nodes fixed
    pos = nx.spring_layout(G, pos=block_pos, fixed=block_nodes, seed=42, weight='weight')

    plt.figure(figsize=(14, 14))

    # Draw nodes with different colors for words and blocks
    word_nodes = [node for node, data in G.nodes(data=True) if data['type'] == 'word']

    nx.draw_networkx_nodes(G, pos, nodelist=word_nodes, node_color='lightblue', node_size=100, label='Words')
    nx.draw_networkx_nodes(G, pos, nodelist=block_nodes, node_color='orange', node_size=300, label='Blocks')

    # Normalize the edge weights to make them thinner
    edges = G.edges(data=True)

    # Generate a color cycle for different blocks
    colors = itertools.cycle([
        'red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray',
        'yellow', 'cyan', 'magenta', 'lime', 'teal', 'maroon', 'navy', 'olive',
        'aqua', 'fuchsia', 'gold', 'indigo'])
    block_colors = {block: next(colors) for block in block_nodes}

    # Draw edges with different colors for each block
    for block in block_nodes:
        block_edges = [(u, v) for u, v, d in edges if d['block'] == block]
        if block_edges:  # Ensure there are edges to process
            weights = np.array([1 / G[u][v]['weight'] if G[u][v]['weight'] > 0 else 1 for u, v in block_edges])
            if len(weights) > 0 and np.max(weights) != np.min(weights):  # Ensure weights are non-empty and not all the same
                normalized_weights = 0.5 + 2 * (weights - np.min(weights)) / (np.max(weights) - np.min(weights))
            else:
                normalized_weights = np.ones(len(weights))  # Assign a default value if no meaningful normalization is possible
            nx.draw_networkx_edges(G, pos, edgelist=block_edges, width=normalized_weights, edge_color=block_colors[block])

    # Draw labels with a custom color
    nx.draw_networkx_labels(G, pos, labels={node: node for node in word_nodes}, font_size=8, font_color='darkred')

    # Draw block labels with thicker font and a different color
    nx.draw_networkx_labels(G, pos, labels={node: node for node in block_nodes}, font_size=10, font_color='darkblue', font_weight='bold')

    plt.title("Word-Block Graph with Normalized Contrast-Based Distances and Circular Block Layout")
    plt.legend()
    plt.show()

def plot_sentence_contrasts(blocks, contrasts, plateau_point):
    combined_contrasts = sorted(contrasts, key=lambda x: x[1], reverse=True)
    # Plotting the combined contrast scores
    x_values = range(1, len(combined_contrasts) + 1)
    y_values = [score for _, score, _ in combined_contrasts]
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_values, marker='o', linestyle='-', color='b')
    plt.axvline(x=plateau_point, color='r', linestyle='--', label='Plateau Point')
    plt.title("Combined Sentence Contrast Scores with Plateau Point")
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

# Generate summary and get the contrasts using preprocessed data
summary, summary_contrast, plateau_point, contrasts = generate_summary(sentence_mappings, preprocessed_blocks)


print("Generated Summary:")
print(summary)

print("\nNumber of sentences selected based on the combined plateau detection:")
print(plateau_point)

plot_sentence_contrasts(sentence_mappings, contrasts, plateau_point)

G = create_word_graph(sentence_mappings, summary_contrast)
plot_word_graph(G)