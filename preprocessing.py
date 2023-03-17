import pandas as pd
import numpy as np
from transformers import GPT2Tokenizer
import xml.etree.ElementTree as ET
import re
from tqdm import tqdm


# Load the dataset into a Pandas dataframe
df = pd.read_csv("your_data_file.tsv", sep="\t")

# Drop the irrelevant columns
df = df[["title", "description"]]

# Clean the text data
def clean_text(text):
    # Remove unwanted characters and punctuations
    text = re.sub(r'[^\w\s]','',text)
    # Convert to lowercase
    text = text.lower()
    # Remove digits
    text = re.sub(r'\d+', '', text)
    return text

df["title"] = df["title"].apply(clean_text)
df["description"] = df["description"].apply(clean_text)

# Tokenize the text data
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

def tokenize_text(text):
    tokens = tokenizer.encode(text, add_special_tokens=True)
    return tokens

df["title_tokens"] = df["title"].apply(tokenize_text)
df["description_tokens"] = df["description"].apply(tokenize_text)

# Encode the tokenized sequences
def encode_tokens(tokens):
    return np.array(tokens)

df["title_encoded"] = df["title_tokens"].apply(encode_tokens)
df["description_encoded"] = df["description_tokens"].apply(encode_tokens)

# Pad or truncate the encoded sequences
max_seq_length = 256

def pad_or_truncate(encoded_seq):
    if len(encoded_seq) > max_seq_length:
        return encoded_seq[:max_seq_length]
    else:
        return np.pad(encoded_seq, (0, max_seq_length - len(encoded_seq)), mode="constant")

df["title_encoded"] = df["title_encoded"].apply(pad_or_truncate)
df["description_encoded"] = df["description_encoded"].apply(pad_or_truncate)

# Save the preprocessed data
preprocessed_data = pd.concat([df["title_encoded"], df["description_encoded"]], axis=1)
preprocessed_data.to_csv("preprocessed_data.tsv", sep="\t", index=False)



# Define a function to preprocess a single post
def preprocess_post(post):
    # Remove code blocks
    post = re.sub(r'<code>.*?</code>', '', post)
    # Remove HTML tags
    post = re.sub(r'<[^>]+>', '', post)
    # Remove URLs
    post = re.sub(r'http\S+', '', post)
    # Remove non-alphanumeric characters
    post = re.sub(r'\W+', ' ', post)
    # Convert to lowercase
    post = post.lower()
    return post.strip()

# Parse the Stack Overflow dataset
tree = ET.parse('stackoverflow-2019.xml')
root = tree.getroot()

# Preprocess and save the posts to a file
with open('stackoverflow_preprocessed.txt', 'w', encoding='utf-8') as f:
    for post in tqdm(root.iter('row'), desc='Preprocessing'):
        if 'Tags' in post.attrib and 'Title' in post.attrib and 'Body' in post.attrib:
            tags = post.attrib['Tags'].replace('><', ',').replace('<', '').replace('>', '').split(',')
            title = preprocess_post(post.attrib['Title'])
            body = preprocess_post(post.attrib['Body'])
            if len(tags) > 0 and len(title) > 0 and len(body) > 0:
                text = ' '.join(tags) + ' ' + title + ' ' + body
                f.write(text + '\n')
