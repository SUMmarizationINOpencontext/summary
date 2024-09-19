import re
import html
import json
from datasets import load_dataset
from bs4 import BeautifulSoup

def clean_text(xml_content):
    """
    Remove XML tags, references, and clean the text content.
    
    Parameters:
        xml_content (list of str): A list of XML strings.
        
    Returns:
        str: A cleaned string of text without XML tags, references, or special characters.
    """
    if not xml_content:
        return ""  # Return empty string if the content is None or empty

    # Join the list of XML content into one string
    joined_content = ' '.join(xml_content)

    # Use BeautifulSoup to remove XML tags and clean the text
    soup = BeautifulSoup(joined_content, "lxml")
    cleaned_text = soup.get_text(separator=" ")

    # Unescape HTML special characters (e.g., &amp; -> &, &lt; -> <)
    cleaned_text = html.unescape(cleaned_text)

    # Remove unwanted references like ##KEYWORD##IDX_REF##OLD_TEXT##
    cleaned_text = re.sub(r'##.*?##', '', cleaned_text)

    # Remove extra line breaks and whitespace
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

    return cleaned_text

def is_valid_pmid(pmid):
    """
    Check if the PMID is a valid number and greater than zero.
    
    Parameters:
        pmid (str or int): The PMID to check.
    
    Returns:
        bool: True if valid, False otherwise.
    """
    try:
        pmid_int = int(pmid)
        return pmid_int > 0
    except ValueError:
        return False

def parse_article(article):
    """
    Parse the article and extract the required sections.
    
    Parameters:
        article (dict): A dictionary containing article data from the dataset.
        
    Returns:
        dict: A dictionary with cleaned text from introduction, methods, results, discussion, and conclusion.
    """
    parsed_article = {
        'pmid': article.get('pmid', 'Unknown'),
        'introduction': clean_text(article.get('introduction', [])),
        'methods': clean_text(article.get('methods', [])),
        'results': clean_text(article.get('results', [])),
        'discussion': clean_text(article.get('discussion', [])),
        'conclusion': clean_text(article.get('conclusion', []))
    }

    # Count how many sections are non-empty
    non_empty_sections = sum(bool(parsed_article[section]) for section in ['introduction', 'methods', 'results', 'discussion', 'conclusion'])

    # Check if the PMID is valid and if there are at least 3 non-empty sections
    if not is_valid_pmid(parsed_article['pmid']) or non_empty_sections < 3:
        return None  # Skip this article if the pmid is invalid or sections are fewer than 3

    return parsed_article

def process_dataset_to_json(dataset, output_file, limit=1000):
    """
    Process the dataset by taking a specified number of articles and saving them to a JSON file.
    
    Parameters:
        dataset (IterableDataset): The dataset in streaming mode.
        output_file (str): Path to the output JSON file.
        limit (int): The number of articles to process.
    """
    articles = {}
    for i, article in enumerate(dataset.take(limit)):
        parsed_article = parse_article(article)
        if parsed_article:
            pmid = parsed_article['pmid']
            articles[pmid] = parsed_article
            print(f"Processed Article {i+1} - PMID: {pmid}")

    # Save the processed articles to a JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(articles, f, indent=4)

    print(f"\nProcessed and saved {len(articles)} articles to {output_file}.")

# Load the dataset in streaming mode (data is streamed without being downloaded)
dataset = load_dataset('TomTBT/pmc_open_access_xml', "commercial", split='train', streaming=True)

# Process the dataset and save to JSON file, taking at most 1000 samples
output_json_file = "processed_articles.json"
process_dataset_to_json(dataset, output_file=output_json_file, limit=2000)