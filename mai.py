import requests
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from textstat import textstat
import pandas as pd
import os
import re

# Ensure NLTK resources are downloaded
nltk.download('stopwords')

# Load input file
input_path = 'Input.xlsx'
input_data = pd.read_excel(input_path)
output_path = 'Output Data Structure.xlsx'

# Load output template for structure
output_data = pd.read_excel(output_path)
output_columns = output_data.columns

# Stopwords for text cleaning
stop_words = set(stopwords.words('english'))


# Helper functions
def extract_article(url):
    """Extract article title and text from the URL."""
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        title = soup.find('title').get_text()
        article_text = ''.join([p.get_text() for p in soup.find_all('p')])
        return title, article_text
    except Exception as e:
        print(f"Error extracting article from {url}: {e}")
        return '', ''


def compute_text_metrics(text):
    """Compute various text metrics as per the requirements."""
    words = text.split()
    word_count = len(words)
    sentence_count = text.count('.')
    complex_words = [word for word in words if len(word) > 2 and textstat.syllable_count(word) > 2]
    complex_word_count = len(complex_words)
    syllables_per_word = sum([textstat.syllable_count(word) for word in words]) / word_count
    avg_word_length = sum(len(word) for word in words) / word_count

    positive_score = sum(1 for word in words if word.lower() in stop_words)  # Example positive list
    negative_score = sum(1 for word in words if word.lower() in stop_words)  # Example negative list

    polarity_score = (positive_score - negative_score) / ((positive_score + negative_score) + 1e-10)
    subjectivity_score = (positive_score + negative_score) / (word_count + 1e-10)
    avg_sentence_length = word_count / (sentence_count + 1e-10)
    percentage_complex_words = complex_word_count / (word_count + 1e-10) * 100
    fog_index = 0.4 * (avg_sentence_length + percentage_complex_words)

    personal_pronouns = len(re.findall(r'\b(I|we|my|ours|us)\b', text, re.I))

    return [
        positive_score, negative_score, polarity_score, subjectivity_score, avg_sentence_length,
        percentage_complex_words, fog_index, word_count, complex_word_count, syllables_per_word,
        personal_pronouns, avg_word_length
    ]


# Process each URL
results = []
for _, row in input_data.iterrows():
    url_id = row['URL_ID']
    url = row['URL']
    title, text = extract_article(url)

    if text:
        metrics = compute_text_metrics(text)
        results.append([url_id, url, title, text[:100]] + metrics)  # Adding sliced text for context
    else:
        print(f"Skipped processing URL ID {url_id} due to empty text.")

# Create output DataFrame
final_columns = list(output_columns[:4]) + [
    "POSITIVE SCORE", "NEGATIVE SCORE", "POLARITY SCORE", "SUBJECTIVITY SCORE", "AVG SENTENCE LENGTH",
    "PERCENTAGE OF COMPLEX WORDS", "FOG INDEX", "WORD COUNT", "COMPLEX WORD COUNT", "SYLLABLE PER WORD",
    "PERSONAL PRONOUNS", "AVG WORD LENGTH"
]
output_df = pd.DataFrame(results, columns=final_columns)

# Save output to file
output_file = 'Processed_Output.xlsx'
output_df.to_excel(output_file, index=False)
print(f"Output saved to {output_file}")
