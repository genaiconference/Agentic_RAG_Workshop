import base64
import logging
import mimetypes
import os
import pickle
from mimetypes import guess_type
import ast
import re
import tiktoken
from rapidfuzz import fuzz
from collections import defaultdict, Counter
from langchain_text_splitters import MarkdownHeaderTextSplitter
import fitz
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import DocumentAnalysisFeature
from azure.core.credentials import AzureKeyCredential
from azure.core.pipeline.transport import RequestsTransport
from langchain.docstore.document import Document
from langchain.schema.messages import AIMessage, HumanMessage
from PIL import Image

logger = logging.getLogger(__name__)


def crop_image_from_image(image_path, page_number, bounding_box):
    """
    Crops an image based on a bounding box.

    :param image_path: Path to the image file.
    :param page_number: The page number of the image to crop (for TIFF format).
    :param bounding_box: A tuple of (left, upper, right, lower) coordinates for the bounding box.
    :return: A cropped image.
    :rtype: PIL.Image.Image
    """
    with Image.open(image_path) as img:
        if img.format == "TIFF":
            # Open the TIFF image
            img.seek(page_number)
            img = img.copy()

        # The bounding box is expected to be in the format (left, upper, right, lower).
        cropped_image = img.crop(bounding_box)
        return cropped_image


def crop_image_from_pdf_page(pdf_path, page_number, bounding_box):
    """
    Crops a region from a given page in a PDF and returns it as an image.

    :param pdf_path: Path to the PDF file.
    :param page_number: The page number to crop from (0-indexed).
    :param bounding_box: A tuple of (x0, y0, x1, y1) coordinates for the bounding box.
    :return: A PIL Image of the cropped area.
    """
    logger.info("Opening PDF file: %s", pdf_path)

    doc = fitz.open(pdf_path)
    page = doc.load_page(page_number)

    # Cropping the page. The rect requires the coordinates in the format (x0, y0, x1, y1).
    bbx = [x * 72 for x in bounding_box]
    rect = fitz.Rect(bbx)
    pix = page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72), clip=rect)

    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    doc.close()

    return img


def crop_image_from_file(file_path, page_number, bounding_box):
    """
    Crop an image from a file.

    Args:
        file_path (str): The path to the file.
        page_number (int): The page number (for PDF and TIFF files, 0-indexed).
        bounding_box (tuple): The bounding box coordinates in the format (x0, y0, x1, y1).

    Returns:
        A PIL Image of the cropped area.
    """
    mime_type = mimetypes.guess_type(file_path)[0]

    if mime_type == "application/pdf":
        return crop_image_from_pdf_page(file_path, page_number, bounding_box)
    else:
        return crop_image_from_image(file_path, page_number, bounding_box)


# Function to encode a local image into data URL
def local_image_to_data_url(image_path):
    # Guess the MIME type of the image based on the file extension
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = "application/octet-stream"  # Default MIME type if none is found

    # Read and encode the image file
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode("utf-8")

    # Construct the data URL
    return f"data:{mime_type};base64,{base64_encoded_data}"


def encode_image(image_path):
    """Getting the base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def image_summarize(img_base64, prompt, llm):
    """Make image summary"""

    try:
        msg = llm.invoke(
            [
                AIMessage(
                    content="You are an useful & intelligent bot who is very good at image reading and rich interpretation."
                ),
                HumanMessage(
                    content=[
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_base64}"
                            },
                        },
                    ]
                ),
            ]
        )
        return msg.content

    except Exception as e:
        if "ResponsibleAIPolicyViolation" in str(e):
            print("Filtered by content policy — skipping image.")
            return ""  # or return some neutral fallback
        else:
            raise


def understand_image_with_gptv(file, llm):
    """
    Generate summaries and base64 encoded strings for images
    path: Path to list of .jpg files extracted by Unstructured
    """

    prompt = """You are an AI assistant tasked with transcribing and interpreting images.

        The image can be:
        Table, Flowchart, Smart Art, graph, infographics, process maps, decision trees, or a Semi Structured picture.

        Your focus should only be on above such images. You should exclude below:
        Static icons, Company logos, branding elements, watermarks, decorative visuals, illustative pictures, human figures, workplace settings, background images, stock photos, table of content, or any non-informational graphics

        Give a detailed interpretation of only relevant image along with a rich summary by ensuring no detail is missed.
        DO NOT miss out on any specific details.
        DO NOT use hash (example - #, ##, ###) in the image description
        DO NOT interpret/generate output for irrelevant images. Give blank output for them.
        DO NOT make any reference to the image or table or flowchart while answering. Example: 'The image depicts'. Answer in present tense - write as if explaining the underlying concept or framework directly..
        You can also use the image caption to get some idea on what exactly the image is about. If the caption is not provided, ignore it.
        FYI the image caption {caption}

        Note: 
            - Describe **all legends**, symbols, color-based groupings, arrows, hierarchies, and layout-based relationships — even if it's visual, explain what they mean, not how they look.
            - Ensure no element is missed. Capture all text, categories, differentiators
            - Clearly identify and describe all levels of categorization, relationships, placement, and hierarchy between elements—such as rows, columns, groupings, axes, zones, arrows, and directional flows.
            - Fully interpret legends, labels, axes, and groupings without referencing visual cues like color, size, or shape. Instead, explain what they signify in context. Avoid phrases like “described in blue,” “in purple,” or “color-coded etc.”
            - Include any annotations, axes titles, directional indicators (e.g., long-term vs. short-term, and implementation suggestions.
            - Please avoid using language that could be interpreted as inappropriate or harmful.
            """

    base64_image = encode_image(file)
    interpretation = image_summarize(base64_image, prompt, llm)
    return interpretation


def analyze_layout(
    input_file_path: str, doc_intelligence_endpoint: str, doc_intelligence_key: str
):
    """
    Analyzes the layout of a document and extracts figures along with their descriptions, then update the markdown output with the new description.

    Args:
        input_file_path (str): The path to the input document file.
        pages (str): No of pages to be processed.

    Returns:
        str: The updated Markdown content with figure descriptions.

    """
    transport = RequestsTransport(verify=False)

    document_intelligence_client = DocumentIntelligenceClient(
        endpoint=doc_intelligence_endpoint,
        credential=AzureKeyCredential(doc_intelligence_key),
        transport=transport,
        headers={"x-ms-useragent": "sample-code-figure-understanding/1.0.0"},
    )

    with open(input_file_path, "rb") as f:
        poller = document_intelligence_client.begin_analyze_document(
            "prebuilt-layout",
            body=f,
            features=[DocumentAnalysisFeature.FORMULAS],
            content_type="application/octet-stream",
            output_content_format="markdown",
            # pages=pages
        )

    result = poller.result()
    return result


def process_images(input_file_path, output_folder, result, llm):
    image_chunks_list = []

    if result.figures:
        logger.debug("starting up...")
        for idx, figure in enumerate(result.figures):
            if figure["boundingRegions"]:
                logger.debug("entered...")
                caption_region = figure["boundingRegions"]
                caption_content = figure.caption.content if figure.caption else ""
                logger.debug(f"\tCaption: {caption_content}")

                for region in figure["boundingRegions"]:
                    logger.debug("page numer:", region.page_number)

                    if region in caption_region:
                        boundingbox = (
                            region.polygon[0],  # x0 (left)
                            region.polygon[1],  # y0 (top)
                            region.polygon[4],  # x1 (right)
                            region.polygon[5],  # y1 (bottom)
                        )
                        logger.debug(boundingbox)
                        cropped_image = crop_image_from_file(
                            input_file_path, region.page_number - 1, boundingbox
                        )  # page_number is 1-indexed
                        logger.debug("image cropped")
                        # Get the base name of the file
                        base_name = os.path.basename(input_file_path)

                        # Remove the file extension
                        file_name_without_extension = os.path.splitext(base_name)[0]

                        output_file = (
                            f"{file_name_without_extension}_cropped_image_{idx}.png"
                        )

                        if not os.path.exists(output_folder):
                            os.mkdir(output_folder)
                        cropped_image_filename = output_folder + output_file
                        cropped_image.save(cropped_image_filename)
                        logger.info("image saved as" + cropped_image_filename)

                        img_description = understand_image_with_gptv(
                            cropped_image_filename, llm
                        )
                        if img_description:
                            offset = (
                                figure["spans"][0]["offset"]
                                if figure.get("spans")
                                else None
                            )
                            image_chunk = Document(
                                # page_content="Caption:" + figure.caption.content + "/n" + img_description,
                                page_content=img_description,
                                metadata={
                                    "page_number": region.page_number,
                                    "offset": offset,
                                },
                            )
                            image_chunks_list.append(image_chunk)
    return image_chunks_list


def count_tokens(text_content):
    # Load the tokenizer for a specific model (e.g., GPT-4)
    encoding = tiktoken.encoding_for_model("gpt-4")

    # Encode the content to tokenize it
    tokens = encoding.encode(text_content)

    # Output the number of tokens
    #print(f"Number of tokens: {len(tokens)}")
    return len(tokens)
  

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

def normalize_latex_formula(match):
    """
    Clean LaTeX-style inline formula like $C H _ { 4 }$ → ch4
    """
    formula = match.group(1)
    # Remove spaces and LaTeX formatting
    formula = formula.replace(' ', '')
    formula = re.sub(r'_?\{?(\d+)\}?', r'\1', formula)  # remove _{4} → 4
    return formula.lower()

def create_chunks_new_v3(result,result_with_image_descp, llm):  #-------> current approach
    # Initialize the MarkdownHeaderTextSplitter with custom headers
    parent_headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2")
    ]

    final_result = dynamic_markdown_split(
    content=result_with_image_descp,
    initial_headers=parent_headers_to_split_on #<should get it from pipeline.xlsx>
)

    page_splits = final_result['chunks']
    md_header_splits = []
    paragraph_positions = paragraphs_with_page_numbers(result)

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
    return md_header_splits , final_result['headers_used']

def dynamic_markdown_split(content, initial_headers, token_limit=3000):
    current_config = initial_headers.copy()
    max_header_level = 4  # Maximum allowed markdown header depth (######)

    while True:
        # Create splitter with current configuration
        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=current_config,
            strip_headers=True
        )

        # Split the document
        split_docs = splitter.split_text(content)

        # Check if all chunks meet token requirements
        if all(count_tokens(doc.page_content) <= token_limit for doc in split_docs):
            return {
                'chunks': split_docs,
                'headers_used': current_config,
                'status': 'success'
            }

        # Exit if we've reached maximum header depth
        if not current_config or len(current_config[-1][0]) >= max_header_level:
            return {
                'chunks': split_docs,
                'headers_used': current_config,
                'status': 'warning: could not meet token limit'
            }

        # Generate next header level (add one more '#' to the last header)
        last_header_symbol, last_header_name = current_config[-1]
        next_level = len(last_header_symbol) + 1
        current_config = current_config.copy() + [
            ('#' * next_level, f'Header_{next_level}')
        ]

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


def create_custom_metadata_for_all_sources(item):  
   # item.metadata['source'] = source  
      
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


def insert_figures_into_full_text(span_map: list, result, image_chunks: list):
  inserted_text = result.content

  # If figures is None, replace with empty list
  if not result.figures:
      return inserted_text  # No figures to insert

  # Create a dict mapping from offset to fig_description from image_chunks
  offset_to_description = {
      chunk.metadata.get("offset"): chunk.page_content
      for chunk in image_chunks
      if chunk.metadata.get("offset") is not None
  }

  # Sort figures by offset descending to avoid offset shifting during insertion
  for fig in sorted(
      result.figures, key=lambda f: f["spans"][0]["offset"], reverse=True
  ):
      fig_offset = fig["spans"][0]["offset"]

      # Skip if we don't have a description for this figure's offset
      if fig_offset not in offset_to_description:
          continue

      fig_description = offset_to_description[fig_offset]

      # Find the nearest span in span_map to decide where to insert
      closest = min(span_map, key=lambda x: abs(x[0] - fig_offset))
      insert_at = closest[0]

      # Insert fig_description at the appropriate position
      inserted_text = (
          inserted_text[:insert_at]
          + f"\n{fig_description}\n"
          + inserted_text[insert_at:]
      )

  return inserted_text



def analyze_document(input_path, key, endpoint):
    """Analyze document layout using DI service."""
    return analyze_layout(
        input_path,
        doc_intelligence_key=key,
        doc_intelligence_endpoint=endpoint,
    )


def extract_span_map(md_result):
    """Build span map from DI results."""
    pages = getattr(md_result, "pages", [])
    return [
        (word.get("span", {}).get("offset"), word.get("content"))
        for page in pages
        for word in page.get("words", [])
    ]


def save_pickle(data, path):
    """Save data to a pickle file."""
    with open(path, "wb") as f:
        pickle.dump(data, f)
    print(f"[INFO] Pickle saved to {path}")


def generate_parents(md_result, full_text_with_images, llm):
    """Generate parent document chunks and metadata."""
    print("[INFO] Creating parent documents...")
    parent_docs, used_headers = create_chunks_new_v3(md_result, full_text_with_images, llm)

    for doc in parent_docs:
        create_custom_metadata_for_all_sources(doc)

    final_parents = append_custom_metadata(parent_docs)
    print(f"[INFO] Created {len(final_parents)} parent documents.")
    return final_parents

