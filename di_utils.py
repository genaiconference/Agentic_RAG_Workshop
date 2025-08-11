import base64
import logging
import mimetypes
import os
from mimetypes import guess_type
import fitz
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import DocumentAnalysisFeature
from azure.core.credentials import AzureKeyCredential
from azure.core.pipeline.transport import RequestsTransport
from langchain.docstore.document import Document
from langchain.schema.messages import AIMessage, HumanMessage
from PIL import Image


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


def analyze_document_layout(
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
        print("starting up...")
        for idx, figure in enumerate(result.figures):
            if figure["boundingRegions"]:
                print("entered...")
                caption_region = figure["boundingRegions"]
                caption_content = figure.caption.content if figure.caption else ""
                print(f"\tCaption: {caption_content}")

                for region in figure["boundingRegions"]:
                    print("page numer:", region.page_number)

                    if region in caption_region:
                        boundingbox = (
                            region.polygon[0],  # x0 (left)
                            region.polygon[1],  # y0 (top)
                            region.polygon[4],  # x1 (right)
                            region.polygon[5],  # y1 (bottom)
                        )
                        #print(boundingbox)
                        cropped_image = crop_image_from_file(
                            input_file_path, region.page_number - 1, boundingbox
                        )  # page_number is 1-indexed
                        print("image cropped")
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
                        print("image saved as" + cropped_image_filename)

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


def extract_span_map(md_result):
    """Build span map from DI results."""
    pages = getattr(md_result, "pages", [])
    return [
        (word.get("span", {}).get("offset"), word.get("content"))
        for page in pages
        for word in page.get("words", [])
    ]