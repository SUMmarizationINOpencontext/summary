<a id="summary_updated"></a>

# summary\_updated

<a id="summary_updated.ExtractiveSummary"></a>

## ExtractiveSummary Objects

```python
class ExtractiveSummary()
```

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

<a id="summary_updated.ExtractiveSummary.preprocess_block"></a>

#### preprocess\_block

```python
def preprocess_block(block)
```

Preprocesses a block of text by tokenizing sentences and words, removing stop words,
and lemmatizing the tokens. It also calculates word frequencies.

**Arguments**:

- `block` _list of str_ - A list of paragraphs to be processed.
  

**Returns**:

- `tuple` - A tuple containing:
  - word_frequencies (dict): A dictionary with words as keys and their frequencies as values.
  - sentence_to_words (list of tuples): A list of tuples where each tuple contains a sentence
  and a list of processed tokens from that sentence.

<a id="summary_updated.ExtractiveSummary.compute_f_measure"></a>

#### compute\_f\_measure

```python
def compute_f_measure(block_frequencies)
```

Compute the F-measure for each word in the given block frequencies.

The F-measure is calculated as the harmonic mean of recall and predominance.
Recall is the frequency of the word in the current block divided by the total
frequency of all words in the block. Predominance is the frequency of the word
in the current block divided by its total frequency across all blocks.

**Arguments**:

- `block_frequencies` _dict_ - A dictionary where keys are words and values are
  their frequencies in the current block.
  

**Returns**:

- `dict` - A dictionary where keys are words and values are their computed F-measures.

<a id="summary_updated.ExtractiveSummary.calculate_FF_bar"></a>

#### calculate\_FF\_bar

```python
def calculate_FF_bar(FF)
```

Calculate the mean of the given FF values.

**Arguments**:

- `FF` _pandas.Series or numpy.ndarray_ - A series or array of FF values.
  

**Returns**:

- `float` - The mean of the FF values.

<a id="summary_updated.ExtractiveSummary.calculate_contrast"></a>

#### calculate\_contrast

```python
def calculate_contrast(FF_block)
```

Calculate the contrast of the given FF_block.

This method divides each element of the FF_block by the corresponding element
in self.FF_bar along the specified axis and fills any resulting NaN values with 0.

**Arguments**:

- `FF_block` _pd.DataFrame_ - The input DataFrame for which the contrast is to be calculated.
  

**Returns**:

- `pd.DataFrame` - A DataFrame with the contrast values.

<a id="summary_updated.ExtractiveSummary.compute_contrast_for_sentences"></a>

#### compute\_contrast\_for\_sentences

```python
def compute_contrast_for_sentences()
```

Computes the contrast for sentences within each block and appends the results to the contrasts list.

This method iterates over the sentence mappings, calculates the F-measure for each block, filters the F-measure
values based on a predefined threshold, and then calculates the contrast for each block. For each sentence in the
block, it computes a sentence score based on the contrast values of the words in the sentence and appends the
sentence, its score, and the block name to the contrasts list.

**Returns**:

  None

<a id="summary_updated.ExtractiveSummary.find_plateau_point_with_tolerance"></a>

#### find\_plateau\_point\_with\_tolerance

```python
def find_plateau_point_with_tolerance(window_size=5, tolerance=0.1)
```

Finds the index of the plateau point in the sorted contrasts list within a given tolerance.

This method sorts the contrasts by their scores in descending order, calculates the differences
between consecutive scores, and then checks for a window of differences where the standard deviation
is less than the specified tolerance. If such a window is found, the method returns the index of the
plateau point.

**Arguments**:

- `window_size` _int, optional_ - The size of the window to check for a plateau. Default is 5.
- `tolerance` _float, optional_ - The tolerance for the standard deviation of the differences within the window. Default is 0.1.
  

**Returns**:

- `int` - The index of the plateau point if found, otherwise the length of the scores list.

<a id="summary_updated.ExtractiveSummary.generate_summary"></a>

#### generate\_summary

```python
def generate_summary()
```

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

**Returns**:

- `tuple` - A tuple containing:
  - final_summary (str): The generated summary as a single string.
  - summary_contrasts (list): A list of sentences ranked by their contrast values.
  - plateau_point (int): The identified plateau point.
  - contrasts (list): A list of all contrast values.

<a id="summary_updated.ExtractiveSummary.rank_sentences_by_contrast"></a>

#### rank\_sentences\_by\_contrast

```python
def rank_sentences_by_contrast(summary_sentences)
```

Ranks sentences by contrast.

This method filters and returns a list of tuples containing sentences, their scores,
and their corresponding blocks from the `contrasts` attribute, but only for those
sentences that are present in the provided `summary_sentences`.

**Arguments**:

- `summary_sentences` _list_ - A list of sentences to be ranked.
  

**Returns**:

- `list` - A list of tuples where each tuple contains a sentence, its score, and its block.

<a id="summary_updated.ExtractiveSummary.create_word_graph"></a>

#### create\_word\_graph

```python
def create_word_graph(summary_contrasts)
```

Creates a word graph from the given summary contrasts.

This function constructs a bipartite graph where one set of nodes represents words and the other set represents blocks (sentences).
Edges between word nodes and block nodes are weighted based on the inverse of the number of blocks the word appears in.

**Arguments**:

- `summary_contrasts` _list of tuples_ - A list of tuples where each tuple contains a sentence, its contrast, and the block name.
  

**Returns**:

- `networkx.Graph` - A bipartite graph with word nodes and block nodes.

<a id="summary_updated.ExtractiveSummary.plot_word_graph"></a>

#### plot\_word\_graph

```python
def plot_word_graph(G)
```

Plots a word-block graph with normalized contrast-based distances and a circular block layout.

**Arguments**:

  -----------
  G : networkx.Graph
  The graph to be plotted. Nodes should have a 'type' attribute with values 'word' or 'block'.
  Edges should have a 'weight' attribute and a 'block' attribute indicating the block they belong to.
  

**Notes**:

  ------
  - Block nodes are arranged in a circular layout.
  - Word nodes are positioned using a spring layout with block nodes fixed.
  - Edges are colored based on the block they belong to.
  - Edge widths are normalized based on their weights.
  - Word nodes are labeled in dark red, and block nodes are labeled in dark blue with bold font.
  - A legend and title are added to the plot.

<a id="summary_updated.ExtractiveSummary.plot_sentence_contrasts"></a>

#### plot\_sentence\_contrasts

```python
def plot_sentence_contrasts(contrasts, plateau_point)
```

Plots the combined sentence contrast scores with a specified plateau point.

**Arguments**:

- `contrasts` _list of tuples_ - A list of tuples where each tuple contains
  (sentence, contrast_score, additional_info). The list is sorted in
  descending order based on the contrast_score.
- `plateau_point` _int_ - The index at which the plateau point is marked on the plot.
  

**Returns**:

  None

