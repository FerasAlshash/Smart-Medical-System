import requests
from PIL import Image
from langchain_mistralai.chat_models import ChatMistralAI
from langchain.prompts.chat import ChatPromptTemplate
import io
import fitz  # PyMuPDF

# Initialize the chatbot model and memory
model = ChatMistralAI(
    mistral_api_key="22MIvQ0KgV5u0ZvxZdecv5wmwAwWhMPZ",
    model_name="pixtral-12b-2409"
)

def extract_text_from_pdf(pdf_content):
    """Extract text from a PDF file."""
    doc = fitz.open(stream=pdf_content, filetype="pdf")
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

def extract_text_from_image(image_content):
    """Extract text from an image file using OCR.space API."""
    api_key = 'K84532933488957'
    url = 'https://api.ocr.space/parse/image'
    payload = {
        'apikey': api_key,
        'language': 'eng',
        'isOverlayRequired': False
    }
    files = {'file': ('image.jpg', image_content)}
    response = requests.post(url, files=files, data=payload)
    result = response.json()

    if 'ParsedResults' in result and len(result['ParsedResults']) > 0:
        return result['ParsedResults'][0]['ParsedText']
    else:
        return "No text detected."

def analyze_document(file):
    """Analyze the uploaded document and provide health recommendations."""
    file_content = file.read()
    file_extension = file.filename.split('.')[-1].lower()

    if file_extension == 'pdf':
        text = extract_text_from_pdf(file_content)
    elif file_extension in ['png', 'jpg', 'jpeg']:
        text = extract_text_from_image(file_content)
    else:
        return "Unsupported file format. Please upload a PDF or image file."

    # Create a prompt for the chat model
    template = """
    You are a medical assistant AI. Analyze the following medical report and provide a detailed explanation and health recommendations based on the extracted data.

    Report:
    {text}

    Provide the analysis in a structured format, including:
    - Summary of the report
    - Key findings
    - Health recommendations
    - Any warnings or alerts based on the results
    """

    prompt = ChatPromptTemplate.from_template(template)
    formatted_prompt = prompt.format_messages(text=text)

    # Get the response from the model
    response = model.invoke(formatted_prompt)
    return response.content
