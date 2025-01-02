# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import spacy
import networkx as nx
import itertools
import string

nlp = spacy.load("fr_core_news_lg")
nlp.add_pipe("merge_entities")
translator = str.maketrans(string.punctuation + " ", '_' * (len(string.punctuation) + 1))


class ExtractiveSummary:
    """
    A class used to perform extractive summarization on text blocks.

    Attributes
    ----------
    blocks : dict
        A dictionary where keys are block names and values are lists of paragraphs.
    preprocessed_blocks : dict
        A dictionary to store preprocessed word frequencies for each block.
    sentence_mappings : dict
        A dictionary to store sentence to words mappings for each block.
    total_frequencies : defaultdict
        A defaultdict to store total word frequencies across all blocks.
    FF_bar : pd.Series or None
        A pandas Series to store the average F-measure for each word.
    contrasts : list
        A list to store sentence contrasts.

    Methods
    -------
    preprocess_block(block)
        Preprocesses a block of text to extract word frequencies and sentence to words mappings.
    preprocess_all_blocks()
        Preprocesses all blocks of text.
    compute_f_measure(block_frequencies)
        Computes the F-measure for each word in a block.
    calculate_FF_bar(FF)
        Calculates the average F-measure for each word.
    calculate_contrast(FF_block)
        Calculates the contrast for each word in a block.
    compute_contrast_for_sentences()
        Computes the contrast for each sentence in all blocks.
    find_plateau_point_with_tolerance(window_size=5, tolerance=0.1)
        Finds the plateau point in the contrast scores with a given tolerance.
    generate_summary()
        Generates a summary based on the contrast scores.
    rank_sentences_by_contrast(summary_sentences)
        Ranks sentences by their contrast scores.
    create_word_graph(summary_contrasts)
        Creates a word-block graph based on the summary contrasts.
    plot_word_graph(G)
        Plots the word-block graph.
    plot_sentence_contrasts(contrasts, plateau_point)
        Plots the sentence contrast scores with the plateau point.
    """
    def __init__(self, blocks):
        self.blocks = blocks
        self.preprocessed_blocks = {}
        self.sentence_mappings = {}
        self.total_frequencies = defaultdict(int)
        self.FF_bar = None
        self.contrasts = []

    def preprocess_block(self, block):
        """
        Preprocesses a block of text by tokenizing sentences and words, removing stop words, 
        and lemmatizing the tokens. It also calculates word frequencies.

        Args:
            block (list of str): A list of paragraphs to be processed.

        Returns:
            tuple: A tuple containing:
                - word_frequencies (dict): A dictionary with words as keys and their frequencies as values.
                - sentence_to_words (list of tuples): A list of tuples where each tuple contains a sentence 
                  and a list of processed tokens from that sentence.
        """
        sentence_to_words = []
        all_words = []
        for paragraph in block:
            doc = nlp(paragraph)
            for sentence in doc.sents:
                processed_tokens = []
                for token in sentence:
                    if token.ent_type_ != "":
                        processed_tokens.append(token.text.translate(translator).lower())
                    elif not token.is_stop and token.is_alpha and token.ent_type_ == "" and token.pos_ not in ('VERB', 'ADV'):
                        processed_tokens.append(token.lemma_.lower())
                sentence_to_words.append((sentence.text, processed_tokens))
                all_words.extend(processed_tokens)
        word_frequencies = dict(Counter(all_words))
        return word_frequencies, sentence_to_words

    def preprocess_all_blocks(self):
        for block_name, block in self.blocks.items():
            word_frequencies, sentence_to_words = self.preprocess_block(block)
            self.preprocessed_blocks[block_name] = word_frequencies
            self.sentence_mappings[block_name] = sentence_to_words

    def compute_f_measure(self, block_frequencies):
        """
        Compute the F-measure for each word in the given block frequencies.

        The F-measure is calculated as the harmonic mean of recall and predominance.
        Recall is the frequency of the word in the current block divided by the total
        frequency of all words in the block. Predominance is the frequency of the word
        in the current block divided by its total frequency across all blocks.

        Args:
            block_frequencies (dict): A dictionary where keys are words and values are
                                      their frequencies in the current block.

        Returns:
            dict: A dictionary where keys are words and values are their computed F-measures.
        """
        f_measures = {}
        total_sum = sum(block_frequencies.values())
        for word, count in block_frequencies.items():
            recall = count / total_sum
            predominance = count / self.total_frequencies[word]
            f_measure = 2 * (recall * predominance) / (recall + predominance) if recall + predominance > 0 else 0
            f_measures[word] = f_measure
        return f_measures

    def calculate_FF_bar(self, FF):
        """
        Calculate the mean of the given FF values.

        Parameters:
        FF (pandas.Series or numpy.ndarray): A series or array of FF values.

        Returns:
        float: The mean of the FF values.
        """
        return FF.mean()

    def calculate_contrast(self, FF_block):
        """
        Calculate the contrast of the given FF_block.

        This method divides each element of the FF_block by the corresponding element
        in self.FF_bar along the specified axis and fills any resulting NaN values with 0.

        Parameters:
        FF_block (pd.DataFrame): The input DataFrame for which the contrast is to be calculated.

        Returns:
        pd.DataFrame: A DataFrame with the contrast values.
        """
        return FF_block.div(self.FF_bar, axis=0).fillna(0)

    def compute_contrast_for_sentences(self):
        """
        Computes the contrast for sentences within each block and appends the results to the contrasts list.

        This method iterates over the sentence mappings, calculates the F-measure for each block, filters the F-measure
        values based on a predefined threshold, and then calculates the contrast for each block. For each sentence in the
        block, it computes a sentence score based on the contrast values of the words in the sentence and appends the
        sentence, its score, and the block name to the contrasts list.

        Returns:
            None
        """
        for block_name, sentence_to_words in self.sentence_mappings.items():
            block_f_measure = self.compute_f_measure(self.preprocessed_blocks[block_name])
            block_f_measure_filtered = {word: f_measure for word, f_measure in block_f_measure.items() if f_measure >= self.FF_bar[word]}
            block_ff = pd.Series(block_f_measure_filtered)
            block_contrast = self.calculate_contrast(block_ff)
            for sentence, words in sentence_to_words:
                sentence_score = sum(block_contrast.get(word, 0) for word in words)
                self.contrasts.append((sentence, sentence_score, block_name))

    def find_plateau_point_with_tolerance(self, window_size=5, tolerance=0.1):
        """
        Finds the index of the plateau point in the sorted contrasts list within a given tolerance.

        This method sorts the contrasts by their scores in descending order, calculates the differences
        between consecutive scores, and then checks for a window of differences where the standard deviation
        is less than the specified tolerance. If such a window is found, the method returns the index of the
        plateau point.

        Args:
            window_size (int, optional): The size of the window to check for a plateau. Default is 5.
            tolerance (float, optional): The tolerance for the standard deviation of the differences within the window. Default is 0.1.

        Returns:
            int: The index of the plateau point if found, otherwise the length of the scores list.
        """
        sorted_contrasts = sorted(self.contrasts, key=lambda x: x[1], reverse=True)
        scores = [score for _, score, _ in sorted_contrasts]
        differences = np.diff(scores)
        for i in range(len(differences) - window_size):
            window = differences[i:i + window_size]
            if np.std(window) < tolerance:
                return i + 2
        return len(scores)

    def generate_summary(self):
        """
        Generates a summary of the text based on preprocessed blocks and their frequencies.

        This method performs the following steps:
        1. Aggregates word frequencies from preprocessed blocks.
        2. Computes F-measures for each block and creates a DataFrame.
        3. Calculates the average F-measure (FF_bar).
        4. Computes contrast values for sentences.
        5. Identifies the plateau point with a specified window size and tolerance.
        6. Sorts sentences by their contrast values.
        7. Selects sentences up to the plateau point to form the summary.
        8. Maps sentences back to their original form and ranks them by contrast.

        Returns:
            tuple: A tuple containing:
                - final_summary (str): The generated summary as a single string.
                - summary_contrasts (list): A list of sentences ranked by their contrast values.
                - plateau_point (int): The identified plateau point.
                - contrasts (list): A list of all contrast values.
        """
        for block_frequencies in self.preprocessed_blocks.values():
            for word, count in block_frequencies.items():
                self.total_frequencies[word] += count
        block_f_measures = [pd.Series(self.compute_f_measure(block_frequencies)) for block_frequencies in self.preprocessed_blocks.values()]
        FF = pd.DataFrame(block_f_measures).fillna(0)
        self.FF_bar = self.calculate_FF_bar(FF)
        self.compute_contrast_for_sentences()
        plateau_point = self.find_plateau_point_with_tolerance(window_size=2, tolerance=1)
        sorted_contrasts = sorted(self.contrasts, key=lambda x: x[1], reverse=True)
        summary_sentences = [sentence for sentence, _, _ in sorted_contrasts[:plateau_point]]
        final_summary = []
        for block_name, sentence_to_words in self.sentence_mappings.items():
            for sentence, words in sentence_to_words:
                if sentence in summary_sentences:
                    final_summary.append(sentence)
        summary_contrasts = self.rank_sentences_by_contrast(summary_sentences)
        return ' '.join(final_summary), summary_contrasts, plateau_point, self.contrasts

    def rank_sentences_by_contrast(self, summary_sentences):
        """
        Ranks sentences by contrast.

        This method filters and returns a list of tuples containing sentences, their scores, 
        and their corresponding blocks from the `contrasts` attribute, but only for those 
        sentences that are present in the provided `summary_sentences`.

        Args:
            summary_sentences (list): A list of sentences to be ranked.

        Returns:
            list: A list of tuples where each tuple contains a sentence, its score, and its block.
        """
        return [(sentence, score, block) for sentence, score, block in self.contrasts if sentence in summary_sentences]

    def create_word_graph(self, summary_contrasts):
        """
        Creates a word graph from the given summary contrasts.

        This function constructs a bipartite graph where one set of nodes represents words and the other set represents blocks (sentences).
        Edges between word nodes and block nodes are weighted based on the inverse of the number of blocks the word appears in.

        Args:
            summary_contrasts (list of tuples): A list of tuples where each tuple contains a sentence, its contrast, and the block name.

        Returns:
            networkx.Graph: A bipartite graph with word nodes and block nodes.
        """
        G = nx.Graph()
        word_nodes = set()
        block_nodes = set(self.sentence_mappings.keys())
        word_to_blocks = defaultdict(set)
        for sentence, contrast, block_name in summary_contrasts:
            preprocessed_words = None
            for sent, words in self.sentence_mappings[block_name]:
                if sent == sentence:
                    preprocessed_words = words
                    break
            if preprocessed_words is None:
                continue
            block_nodes.add(block_name)
            for word in preprocessed_words:
                word_nodes.add(word)
                word_to_blocks[word].add(block_name)
        for block in block_nodes:
            G.add_node(block, type='block')
        for word in word_nodes:
            G.add_node(word, type='word')
            for block in word_to_blocks[word]:
                weight = 1 / len(word_to_blocks[word])
                G.add_edge(word, block, weight=weight, block=block)
        return G

    def plot_word_graph(self, G):
        """
        Plots a word-block graph with normalized contrast-based distances and a circular block layout.

        Parameters:
        -----------
        G : networkx.Graph
            The graph to be plotted. Nodes should have a 'type' attribute with values 'word' or 'block'.
            Edges should have a 'weight' attribute and a 'block' attribute indicating the block they belong to.

        Notes:
        ------
        - Block nodes are arranged in a circular layout.
        - Word nodes are positioned using a spring layout with block nodes fixed.
        - Edges are colored based on the block they belong to.
        - Edge widths are normalized based on their weights.
        - Word nodes are labeled in dark red, and block nodes are labeled in dark blue with bold font.
        - A legend and title are added to the plot.
        """
        block_nodes = [node for node, data in G.nodes(data=True) if data['type'] == 'block']
        num_blocks = len(block_nodes)
        angle_step = 2 * np.pi / num_blocks
        block_pos = {block: (np.cos(i * angle_step), np.sin(i * angle_step)) for i, block in enumerate(block_nodes)}
        pos = nx.spring_layout(G, pos=block_pos, fixed=block_nodes, seed=42, weight='weight')
        plt.figure(figsize=(14, 14))
        word_nodes = [node for node, data in G.nodes(data=True) if data['type'] == 'word']
        nx.draw_networkx_nodes(G, pos, nodelist=word_nodes, node_color='lightblue', node_size=100, label='Words')
        nx.draw_networkx_nodes(G, pos, nodelist=block_nodes, node_color='orange', node_size=300, label='Blocks')
        edges = G.edges(data=True)
        colors = itertools.cycle(['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'yellow', 'cyan', 'magenta', 'lime', 'teal', 'maroon', 'navy', 'olive', 'aqua', 'fuchsia', 'gold', 'indigo'])
        block_colors = {block: next(colors) for block in block_nodes}
        for block in block_nodes:
            block_edges = [(u, v) for u, v, d in edges if d['block'] == block]
            if block_edges:
                weights = np.array([1 / G[u][v]['weight'] if G[u][v]['weight'] > 0 else 1 for u, v in block_edges])
                if len(weights) > 0 and np.max(weights) != np.min(weights):
                    normalized_weights = 0.5 + 2 * (weights - np.min(weights)) / (np.max(weights) - np.min(weights))
                else:
                    normalized_weights = np.ones(len(weights))
                nx.draw_networkx_edges(G, pos, edgelist=block_edges, width=normalized_weights, edge_color=block_colors[block])
        nx.draw_networkx_labels(G, pos, labels={node: node for node in word_nodes}, font_size=8, font_color='darkred')
        nx.draw_networkx_labels(G, pos, labels={node: node for node in block_nodes}, font_size=10, font_color='darkblue', font_weight='bold')
        plt.title("Word-Block Graph with Normalized Contrast-Based Distances and Circular Block Layout")
        plt.legend()
        plt.show()

    def plot_sentence_contrasts(self, contrasts, plateau_point):
        """
        Plots the combined sentence contrast scores with a specified plateau point.

        Args:
            contrasts (list of tuples): A list of tuples where each tuple contains
                (sentence, contrast_score, additional_info). The list is sorted in
                descending order based on the contrast_score.
            plateau_point (int): The index at which the plateau point is marked on the plot.

        Returns:
            None
        """
        combined_contrasts = sorted(contrasts, key=lambda x: x[1], reverse=True)
        x_values = range(1, len(combined_contrasts) + 1)
        y_values = [score for _, score, _ in combined_contrasts]
        plt.figure(figsize=(10, 6))
        plt.plot(x_values, y_values, marker='o', linestyle='-', color='b')
        plt.axvline(x=plateau_point, color='r', linestyle='--', label='Plateau Point')
        plt.title("Combined Sentence Contrast Scores with Plateau Point")
        plt.xlabel("Sentence Index")
        plt.ylabel("Contrast Score")
        plt.legend()
        plt.show()


# Example usage:
# blocks = {
#     "Introduction": "Your text here...",
#     "Material and methods": "Your text here...",
#     # Add more blocks as needed
# }

# summary_generator = ExtractiveSummary(blocks)
# summary_generator.preprocess_all_blocks()
# summary, summary_contrast, plateau_point, contrasts = summary_generator.generate_summary()

# print("Generated Summary:")
# print(summary)

# print("\nNumber of sentences selected based on the combined plateau detection:")
# print(plateau_point)

# G = summary_generator.create_word_graph(summary_contrast)
# summary_generator.plot_word_graph(G)
# summary_generator.plot_sentence_contrasts(contrasts, plateau_point)