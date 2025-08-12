from langchain_text_splitters import MarkdownHeaderTextSplitter
from rapidfuzz import fuzz
import tiktoken
import re
from collections import defaultdict


def create_custom_metadata_for_all_sources(item):
    headers = ['Header 4', 'Header 3', 'Header 2', 'Header 1']  
      
    # Find the first existing header in reverse order  
    for header in headers: 
        if header in item.metadata:  
            item.metadata['custom_metadata'] = item.metadata[header]
            print('')
            break  
      
    return item 


def append_custom_metadata(docs):
    """
    Appends custom metadata to the beginning of the page content for each document in the list.
    Args:
        docs (list): A list of documents with custom metadata in their metadata dictionaries.
    Returns:
        list: A list of documents with updated page content.
    """
    for doc in docs:
        if "custom_metadata" in doc.metadata:
          if doc.metadata['custom_metadata'] == doc.metadata['Header 1']:
            doc.page_content = "#" + doc.metadata['custom_metadata'] + "\n\n" + doc.page_content
          elif doc.metadata['custom_metadata'] == doc.metadata['Header 2']:
            doc.page_content = "##" + doc.metadata['custom_metadata'] + "\n\n " + doc.page_content
          else:
            doc.page_content = doc.metadata['custom_metadata'] + "\n\n" + doc.page_content

    return docs


def add_source_metadata(docs, source_file_name):
  for doc in docs:
    doc.metadata['source'] = source_file_name
  return docs


def count_tokens(text_content):
    # Load the tokenizer for a specific model (e.g., GPT-4)
    encoding = tiktoken.encoding_for_model("gpt-4")

    # Encode the content to tokenize it
    tokens = encoding.encode(text_content)

    # Output the number of tokens
    #print(f"Number of tokens: {len(tokens)}")
    return len(tokens)
  

def normalize_latex_formula(match):
    """
    Clean LaTeX-style inline formula like $C H _ { 4 }$ → ch4
    """
    formula = match.group(1)
    # Remove spaces and LaTeX formatting
    formula = formula.replace(' ', '')
    formula = re.sub(r'_?\{?(\d+)\}?', r'\1', formula)  # remove _{4} → 4
    return formula.lower()
    

def clean_text(text):
    # Handle escaped newlines and bullets
    text = text.replace('\\n', ' ')
    text = text.replace('\n', ' ')
    text = re.sub(r'(\d+)\\\.', r'\1.', text)
    text = text.replace('\\', '')

    # Normalize LaTeX-style formulas: $C H _ { 4 }$ → ch4
    text = re.sub(r'\$([^$]+)\$', normalize_latex_formula, text)

    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def is_mostly_numbers(text):
    # Remove normal punctuations and spaces
    cleaned = re.sub(r'[\s.,()\[\]{}\-–—]', '', text)
    # If after cleaning, it's only digits, return True
    return cleaned.isdigit()


def is_single_word(text):
    # Split the text into words
    words = text.strip().split()
    # Check if there's only one word
    return len(words) == 1


def paragraphs_with_page_numbers(result):
    paragraph_positions = defaultdict(list)

    for paragraph in result.paragraphs:
        if not hasattr(paragraph, 'role') or not paragraph.role:  # role missing
            paragraph_text = clean_text(paragraph.content).lower()
            if paragraph_text and not is_mostly_numbers(paragraph_text) and not is_single_word(paragraph_text):
                paragraph_positions[paragraph_text].append(paragraph.bounding_regions[0]["pageNumber"])
    return paragraph_positions



def assign_page_numbers_to_parent_docs(md_result, page_splits):
    """
    This function does assign page numbers to each parent chunk.
    Using DI there is no direct way to get page numbers.
    Below is the custom logic to find out right pagenumber to each chunk and add it as metadata
    """
    md_header_splits = []
    paragraph_positions = paragraphs_with_page_numbers(md_result)
    for chunk in page_splits:
        chunk_text_cleaned = clean_text(chunk.page_content).lower()
        page_weight_scores = defaultdict(float)

        # Exact matching
        for paragraph_text, page_numbers in paragraph_positions.items():
            if paragraph_text in chunk_text_cleaned:
                para_len = len(paragraph_text)
                chunk_len = len(chunk_text_cleaned)

                if chunk_len == 0:
                    continue

                # Compute weight as ratio of paragraph to chunk
                weight = para_len / chunk_len

                for page in page_numbers:
                    page_weight_scores[page] += weight

        # If no exact matches, try fuzzy matching with weight
        if not page_weight_scores:
            for paragraph_text, page_numbers in paragraph_positions.items():
                score = fuzz.partial_ratio(paragraph_text, chunk_text_cleaned)
                if score > 90:
                    para_len = len(paragraph_text)
                    chunk_len = len(chunk_text_cleaned)

                    if chunk_len == 0:
                        continue

                    weight = (score / 100) * (para_len / chunk_len)

                    for page in page_numbers:
                        page_weight_scores[page] += weight

        # Assign majority page number
        if page_weight_scores:
            # Pick the page with the highest weighted score
            majority_page = max(page_weight_scores.items(), key=lambda x: x[1])[0]
            chunk.metadata['page_number'] = majority_page
        else:
            chunk.metadata['page_number'] = None

        md_header_splits.append(chunk)
    return md_header_splits
    
