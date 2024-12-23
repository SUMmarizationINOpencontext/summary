# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import defaultdict
import spacy
# from nltk.tokenize import word_tokenize, sent_tokenize
# from nltk.corpus import stopwords
# from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
# from nltk.stem import WordNetLemmatizer
# from nltk import ne_chunk, pos_tag
# import nltk
import networkx as nx
import itertools
from collections import Counter
import string

# nltk.download('punkt')        # Tokenizer models for sentence and word tokenization
# nltk.download('stopwords')    # List of stopwords for various languages
# nltk.download('averaged_perceptron_tagger')  # Part-of-speech tagging models
# nltk.download('maxent_ne_chunker')  # Named Entity Recognition chunker
# nltk.download('words')        # Word list used by NER chunker
# nltk.download('wordnet')      # WordNet lexical database used by the lemmatizer
# nltk.download('punkt_tab')

nlp = spacy.load("fr_core_news_lg")
nlp.add_pipe("merge_entities")
translator = str.maketrans(string.punctuation+" ", '_'*(len(string.punctuation)+1))


def preprocess_block(block):
    sentence_to_words = []  # List to store sentences with their preprocessed words
    all_words = []  # List to store all preprocessed words for the block
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
    

def preprocess_all_blocks(blocks):
    preprocessed_blocks = {}
    sentence_mappings = {}
    
    for block_name, block in blocks.items():
        word_frequencies, sentence_to_words = preprocess_block(block)
        preprocessed_blocks[block_name] = word_frequencies
        sentence_mappings[block_name] = sentence_to_words
    
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
    plateau_point = find_plateau_point_with_tolerance(contrasts, window_size=2, tolerance=1)
    
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


# blocks = {
#     "Introduction": """Hymenoptera (which includes bees, wasps, bumble- bees, and hornets) stings are common in adults and children. In the general population, 56.6–94.5% of people report at least one sting in their lives [1]. Reactions of different severity may occur as a result of Hymenoptera stings. The most common are uncomplicated local re- actions, typical of people who are not allergic to insect venom. In Hymenoptera venom allergy (HVA) large local reactions occur most often, with a frequency estimated
# at 2.4% up to 26.4% [2] of the general population. The prevalence of systemic allergic reactions after Hymenop- tera stings ranges from 0.3% up to 7.5% in adults and up to 0.3% in children [3]. In Europe, HVA is the most com- mon cause of severe allergic reactions in adults (48.2%) and the second cause of anaphylaxis in children (20.2%) [4]. The proper treatment, which should provide very high protection from future life-threating systemic reactions in patients with HVA, depends on the correct diagnosis and qualification for venom immunotherapy (VIT).In the entomological picture Hymenoptera species differ between each other as to the leading character- istics as follows: a bee (Apis mellifera) – a brown insect with a moderate number of hairs on the trunk and abdo- men; a wasp (Vespula spp.) – a bee-sized, yellow-black insect with few hairs on the body and without hairs on the abdomen; a bumblebee (Bombus) – larger and more hairy than a bee, with numerous yellow, white or red stripes; a hornet (Vespa) – twice the size of a wasp, with a slightly darker, reddish head and trunk. These are the most popular in the middle part of Europe. The other entomological genus – Polistes spp. are relevant due to allergic reactions in the Mediterranean region, but they have still weak representation in in middle-northern part of Europe [5].
# Beside the classification of the type of reaction and confirmation of an IgE-mediated pathogenesis, the iden- tification of the offending insect is one of the key points to make the right diagnosis [6, 7]. Information on the appearance and behaviour of the insect, retraction of the sting, natural death of the offending insect after the sting, presence of hives or nests in the nearby area, when available, should be documented from each subject, be- cause these data are helpful to guide the diagnosis, and in future management including the selection of VIT [7]. In the follow-up it should be also the key point for the patient’s education on how to avoid subsequent stings.
# The identification of the stinging insect may prove difficult for the patient and his/her family, as well as for the physician, including the allergy specialist, for several reasons. The stinging insect may be overlooked due to its small size and fast movements. Moreover, as the Hyme- noptera insects are similar to each other, knowledge of the Hymenoptera physical features and its behaviours is crucial for the identification. In order to adequately ad- dress the needs for education in the correct identifica- tion of stinging insects, it is necessary to determine the scale of educational deficits in this area. To the best of our knowledge, there are no studies on both children and their parents’ ability to identify stinging insects.
# """,

#     "Material and methods": """The questionnaire-based study was performed in a tertiary paediatric medical centre – the Children’s Uni- versity Hospital in Krakow. All the patients with HVA and their parents were recruited from the Department of Pediatrics, Pulmonology, Allergology and Dermatology. The subjects without HVA and their parents (some of them stung by insects in the past), to serve as control groups, were randomly selected from the patients and their care- givers who were present in the above medical units for medical reasons other than HVA.
# The analysed group consisted of 102 children with confirmed HVA (according to current EAACI guidelines on allergen immunotherapy [8]) with their 102 parents and 98 children without HVA accompanied by their 98 parents.
# A 7-item questionnaire survey, addressed separately to the child and his/her caregiver, consisted of 2 parts. The first part was dedicated to demographic data and the personal history of stings of the child and the accom- panying parent, respectively. The second part consisted of 5 single-choice sub-questions referring to the recogni- tion of insects presented in photos without captions. De- mographics included age, sex, place of residence, current level of education in reference to children and the highest level of education obtained by parents. The sting history contained the questions addressed to the number of in- sect stings in the past, the kind of culprit insects (bee, wasp, hornet, bumblebee, or an unidentified insect) and the approximate dates of such stings for a child and for his/her caregiver, respectively.
# The images were colour pictures depicting the fol- lowing, typical for the region, four different representa- tives of Hymenoptera order insects for identification: a bee, a wasp, a hornet, a bumblebee, and additionally, as a confounder, one representative of Diptera order – hoverflies, which resembles Hymenoptera, but their stings are harmless to humans. The image of each insect was shown as 2 pictures (top and side view) (see the Supplementary file). Questions referring to the pictures were scored as 1 point for the correct identification and 0 points for an incorrect identification, for a total possible score of 5. Each child and his/her parent responded to the questions separately. It took about 10 min to com- plete the questionnaire.
# The study was approved by the Jagiellonian University Ethics Committee (dated 26 Jan 2017/ No.122.6120.14.2017). The study was performed in accor- dance with the ethical standards as laid down in the 1975 Declaration of Helsinki, as revised in 2000. Because the research involved human participants, written informed consent was obtained before enrolment from the legal guardians (parents) of all participants. The data that sup- port the findings of this study are available from the cor- responding author upon reasonable request.
# The Jagiellonian University Medical College support- ed the study through a subsidy for maintaining research potential. This research received no specific grant from any funding agency in the public, commercial, or not-for- profit sectors.""",
    
#     "Statistical analysis": """Distribution of qualitative variables was presented using frequencies and percentages, whereas for quanti- tative variables, means and SDs for normally distributed ones and medians and quartiles otherwise were used. c2 test was used to examine the relationship between two qualitative variables. If at least 20% of cells in the analyzed table had expected frequencies lower than 5 the exact Fisher test for 2x2 tables and Fisher-Freeman- Halton test otherwise were used. Difference in mean age between studied groups was tested using Student’s t-test for independent samples. The difference in distri- bution of other quantitative variables between 2 groups was analysed using the Mann-Whitney test; when the size of the analysed subsample was lower than 30, the exact version of the test was used. Effects with p < 0.05 were treated as statistically significant. IBM SPSS Statis- tics 25 for Windows software was used.""",
#     "Results": """There was no difference in the mean age of children between Hymenoptera venom allergic group and con- trols (10.3 ±3.7 vs. 11.1 ±3.3 years; p > 0.05). Parents of children with HVA were younger than parents from the control dyads (38.9 ±7.1 vs. 41.5 ±8.3 years; p = 0.016). In both groups, most of children were males, and most of parents were females. The majority of participating children in both groups lived in the village. Most of chil- dren attended primary school. Most parents of children
# with HVA had obtained primary or secondary education, contrary to parents from the non-HVA group (Table 1).
# The differences between groups in the percentage of correct insect identification were statistically significant (p < 0.001) (Figure 1). The percentage of persons cor- rectly identifying all kinds of insects was the highest in the group of parents of children with HVA (92.5%) and their children (91.2%). The lowest rate of participants who identified insects correctly was found in the group of children without HVA (78.8%) and their parents (82.4%).
# The differences between insects in the percentage of their correct identification by all study participants were also statistically significant (p < 0.001). The most frequently recognized insect in all groups was the hornet (up to 96.1% in children with HVA), whereas the least identified one was the hoverfly (down to 69.1% in chil- dren without HVA). In each group (children with HVA and their parents, children without HVA and their parents), the percentage of participants correctly identifying the insect varied depending on the kind of insect (p = 0.024, p < 0.001, p < 0.001, p = 0.018, respectively) (Figure 2).
# The percentage of children stung by a bee, wasp and hornet was higher among the children with HVA than in the respective control group (Table 2). The percentage of parents stung by a bee was higher among parents of the children with HVA than in the respective control group. The children with HVA and their parents were mostly stung by a bee, the median number of stings per person was 3. The wasp was the second insect most often sting- ing children with HVA and their parents with the median number of stings per person amounting to 1 and 2, respectively. The median time from the last sting by a bee was shorter among children with HVA than in children without HVA. In parents of children without HVA, the time period from the last bee sting to the questionnaire survey was much longer than in parents of children with HVA (Table 2).""",
    
#     "Children with HVA": """Children with HVA, in comparison to children with- out HVA, were more likely to correctly identify the bee (92.2% vs. 74.5%, p = 0.001), bumblebee (91.2% vs. 76.5%, p = 0.005) and hoverfly (85.1% vs. 69.4%, p = 0.008). The correct identification of the wasp by children with HVA depended on their place of residence and was more com- mon among children living in the village than in the city (95.0% vs. 76.9%, p = 0.018). Among children with HVA who correctly identified the wasp, the time from being stung by this insect to completing the survey was shorter in relation to the group of children with HVA who were unable to correctly identify the wasp (1.3 (Q1–Q3 0.4–2.6) vs. 5.2 (Q1–Q3 2.4–15.6) years, p = 0.03).
# The rate of children with HVA who were stung by a bee, wasp and hornet was higher in comparison to the group of children without HVA (Table 2). The children with HVA stung by hornets were older than children who were not stung by this insect (12.4 ±3.9 vs. 10.0 ±3.5, p = 0.028). Children living in the countryside, regardless of their HVA status, were more often stung by a bee or a hornet than children living in the city (68.4% vs. 32.8%, p < 0.001 and 9.8% vs. 1.5%, p = 0.038, respectively).""",
    
#     "Children without HVA": """The correct identification of a bee and a bumblebee by children without HVA depended on their place of resi- dence and was more common among children living in the city in comparison to children living in the country- side (86.7% vs. 64.2%, p = 0.011, and 88.9% vs. 66.0%, p = 0.008, respectively).
# Most children without HVA who were attending ju- nior high school (70.0%) were stung by a bee. Minority of children without HVA attending primary (29.6%) or high school (42.9%) were stung by this insect. Children with- out HVA stung by a bee or a wasp were older than chil- dren who were not stung by this insect (12.3 ±3.0 vs. 10.5 ±3.2 years, p = 0.005 and 12.3 ±2.9 vs. 10.4 ±3.3 years, p = 0.005, respectively).""",
    
#     "Parents of children with HVA and parents of children without HVA": """The parents of children with HVA, as compared to the parents of children without HVA, were more likely to correctly identify the bee (96.1% vs. 75.5%, p < 0.001), bumblebee (92% vs. 83.7%, p = 0.048) and hoverfly (88% vs. 74.5%, p = 0.015) (Figure 2). Only in the group of par- ents of children without HVA the correct identification of the bee, wasp and bumblebee depended on the parents’ level of education and was more common among par- ents educated at the high school level than parents who completed only the primary and junior high school (58.1% vs. 41.9%, p = 0.035, 57.8% vs. 42.2%, p = 0.007, 56.8% vs. 43.2%, p = 0.04, respectively).
# The parents of children with HVA were more likely to be stung by a bee than the parents of children without HVA (78.4% vs. 58.2%, p = 0.002). Among the parents of children without HVA, those stung by a bee or by an unidentified insect were older than those who were not stung by these insects (43.2 ±8.0 vs. 39.2 ±8.2 years,
# p = 0.018, and 44.2 ±8.6 vs. 40.4 ±8.0 years, p = 0.040, respectively).""",
    
#     "Discussion": """The problem of Hymenoptera venom allergy is clini- cally important regardless of the patient’s age. According to the Centers for Disease Control and Prevention Wide- Ranging Online Data for Epidemiologic Research data- base of the US, documenting all animal-related fatalities between 2008 and 2015, deaths attributable to Hyme- noptera (hornets, wasps, and bees) account for 29.7% of the overall animal-related fatalities and have been steady over the last 20 years [9]. The data from the on- line Network of Severe Allergic Reactions (NORA) suggest that insect venom is an important trigger of anaphylaxis both in children and adults (20.2% and 48.2%, respec- tively) [4]. Preponderance of stinging insects depends on the geographic region [10]. In Europe, 70.6% of anaphy- lactic reactions are caused by stings of wasps followed by those of bees (23.4%) and of hornets (4.1%) [4]. In the analysed populations, the children with HVA and their parents were mainly stung by bees. It is due to the fact that many patients were bee-keepers’ family members.
# As the diagnosis of HVA negatively affects the quality of life [11–13], it probably makes patients more alert and pay more attention to flying insects. May be to the diag- nostic process, HVA groups are better at recognizing the stinging insects. Moreover, the unpleasant experience associated with an allergic reaction to insect venom and the individual’s need to avoid another future reaction contribute to better identification of insects. This may ex- plain a higher rate of correct insect identification among children with HVA and their parents than in the control groups. The identification abilities in a group of children with HVA did not differ significantly in comparison to the group of their parents. Children without HVA were less likely to correctly recognize stinging insects compared to their parents. Unlike in an earlier study [14], adults with- out HVA most often correctly recognized hornets, then wasps and bumblebees, followed by bees and hoverflies.
# Identification of the culprit insect may be difficult, because insects may sting without being seen, they are relatively small, and share similar features with other members in the order, complicating identification of the perpetrator in many instances. Hymenoptera identifica- tion is difficult even for allergy-trained experts [15]. We have put among Hymenoptera species an example of Diptera (hoverfly, Syrphidae) as a confounder which re- sembles the bee or wasp (though it does not belong to Hymenoptera) in order to better assess the participants’ ability to recognize the insects in question.
# The correct identification of a given insect may de- pend on the time since the last sting. However, this rule was not confirmed in reference to the majority of insects analysed in the study, but only in reference to wasps.Interestingly, the relationship between the level of education and insect recognition skill was observed only in the group of parents of children without HVA referring to the bee, wasp and bumblebee identification. An older age and higher level of education among children were factors that should potentially improve the correctness rate in insect identification also in this population, but this has not been noted in our study.
# According to our data, as well as to these already published, the significant percentage of the patients en- counter difficulties in the correct identification of the in- sect that has stung them and caused the allergic reaction [14, 15], hence there is a need for an appropriate educa- tion. Most individuals have little or no formal education in insect identification [15]. As it is suggested in clinical practice, detailed colour photographs may be the best educational tool when educating individuals on salient characteristics of stinging insects and how to avoid them [15]. Compilation of a picture-based educational insect guidebook is simple and can be a useful resource for ed- ucating patients on stinging insect features and avoid- ance strategies, but these materials may not be helpful as a tool when considering testing to identify stinging insect hypersensitivity [15]. In the diagnostic approach an allergist should assess the patient by taking a clinical history, with the emphasis on the severity of symptoms, and by performing venom testing. Because the possibility of making a mistake in subjective data is high, objective medical records together with allergy tests must be per- formed to confirm the allergy. In some of the cases fur- ther diagnostic tests (component-resolved diagnostics, basophil activation test) are needed to correctly identify the allergy-relevant insect.
# Allergen immunotherapy is instituted in these/indi- viduals with insect sting reaction exceeding skin symp- toms and confirmed IgE-mediated venom allergy [7]. In- sect identification should be a part of both the diagnostic and educational process. The correct identification of the allergy-relevant insect is helpful for testing and accurate therapy of venom-allergic patients [6].
# This study has some limitations to be considered. The first limitation concerns the time since being stung. The longer the time, the more probable an erroneous identi- fication of the insect. It should be noted, however, that in the sample studied the time period between the last stinging event and taking the history from the subjects was longer for the children without HVA than from those with HVA. Another limitation is that picture representa- tion of insects displayed colours slightly different than in reality. Despite this, the results of the previous study suggest that for identification purposes individuals are more likely to be successful using detailed photographs rather than actual dried insects [15].""",
    
#     "Conclusions": """Most of people experienced stings by Hymenoptera insects. Even despite potentially life-threatening allergic reactions, some children with HVA and their parents are not able to identify stinging insects correctly. The abil- ity to identify stinging insects depends on the burden of HVA diagnosis, the distinguishing features of the insect (e.g. size), place of residence, the time that has passed since the sting, and the level of institutionalized educa- tion received. These skills, combined with knowledge of the habits and behaviours of individual insect types, can help prevent stinging, subsequently leading to HVA, and should be an integral part of diagnostic and educational process.""",
# }

# blocksz = {"Sentence" : "Adam Smith was a famous economist from Scotland. He earned 100 dollars."}


# # Preprocess all blocks
# preprocessed_blocks, sentence_mappings = preprocess_all_blocks(blocks)

# # Generate summary and get the contrasts using preprocessed data
# summary, summary_contrast, plateau_point, contrasts = generate_summary(sentence_mappings, preprocessed_blocks)

# print("Generated Summary:")
# print(summary)

# print("\nNumber of sentences selected based on the combined plateau detection:")
# print(plateau_point)

# #plot_sentence_contrasts(sentence_mappings, contrasts, plateau_point)

# G = create_word_graph(sentence_mappings, summary_contrast)
# plot_word_graph(G)