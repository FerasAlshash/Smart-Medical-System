import requests
from langchain_mistralai.chat_models import ChatMistralAI
from langchain.prompts.chat import ChatPromptTemplate
import fitz                                                   # PyMuPDF
import numpy as np
import cv2
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize the chatbot model and memory
model = ChatMistralAI(
    mistral_api_key=os.getenv("MISTRAL_API_KEY"),
    model_name="pixtral-12b-2409"
)

def preprocess_image(image_content):                                                    
    """Preprocess an image to improve OCR results for low quality, shadowed, or angled images."""
    # Convert bytes to numpy array
    nparr = np.frombuffer(image_content, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Check if image was loaded properly
    if img is None:
        return image_content  # Return original if loading failed
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding to handle varying lighting conditions and shadows
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    # Perspective correction to handle angled photos
    # Detect edges
    edges = cv2.Canny(binary, 50, 150, apertureSize=3)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # If substantial contours found, attempt perspective correction
    if contours and len(contours) > 0:
        # Find the largest contour by area
        largest_contour = max(contours, key=cv2.contourArea)
        
        # If the contour has sufficient area, apply perspective transform
        if cv2.contourArea(largest_contour) > 1000:  # Minimum area threshold
            # Approximate the contour to a polygon
            epsilon = 0.02 * cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)
            
            # If we have a quadrilateral, apply perspective transform
            if len(approx) == 4:
                # Sort the points for correct ordering
                approx = sorted(approx, key=lambda x: x[0][0])
                left_points = sorted(approx[:2], key=lambda x: x[0][1])
                right_points = sorted(approx[2:], key=lambda x: x[0][1])
                pts1 = np.float32([left_points[0][0], left_points[1][0], 
                                  right_points[0][0], right_points[1][0]])
                
                # Define the new dimensions
                width, height = 800, 1000
                pts2 = np.float32([[0, 0], [0, height], 
                                  [width, 0], [width, height]])
                
                # Apply perspective transform
                matrix = cv2.getPerspectiveTransform(pts1, pts2)
                result = cv2.warpPerspective(binary, matrix, (width, height))
                
                # Apply noise reduction
                result = cv2.GaussianBlur(result, (3, 3), 0)
                
                # Convert back to bytes for OCR API
                _, processed_image = cv2.imencode('.jpg', result)
                return processed_image.tobytes()
    
    # If perspective correction not applied, apply basic enhancements
    # Apply noise reduction
    denoised = cv2.fastNlMeansDenoising(binary, None, 10, 7, 21)
    
    # Increase contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    
    # Convert back to bytes for OCR API
    _, processed_image = cv2.imencode('.jpg', enhanced)
    return processed_image.tobytes()

def extract_text_from_pdf(pdf_content):
    """Extract text from a PDF file."""
    doc = fitz.open(stream=pdf_content, filetype="pdf")
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

def extract_text_from_image(image_content, language='eng'):
    """
    Extract text from an image file using OCR.space API.
    
    Parameters:
    image_content: bytes content of the image
    language: language code ('eng' for English, 'ara' for Arabic, 'ger' for German)
    """
    # Preprocess the image before sending to OCR
    processed_image = preprocess_image(image_content)
    
    api_key = 'K84532933488957'
    url = 'https://api.ocr.space/parse/image'
    
    # Map language codes to OCR.space language codes
    language_mapping = {
        'eng': 'eng',  # English
        'ara': 'ara',  # Arabic
        'ger': 'ger'   # German
    }
    
    # Use the mapped language code or default to English
    ocr_language = language_mapping.get(language, 'eng')
    
    payload = {                                          # set the payload for the OCR API
        'apikey': api_key,
        'language': ocr_language,
        'isOverlayRequired': False,                      
        'detectOrientation': True,                       # Helpful for Arabic texts that might be right-to-left
        'scale': True,                                   # Improves accuracy for different sized texts
        'OCREngine': 2                                   # Engine 2 works better with non-Latin scripts like Arabic
    }
    
    files = {'file': ('image.jpg', processed_image)}
    response = requests.post(url, files=files, data=payload)
    result = response.json()
    
    if 'ParsedResults' in result and len(result['ParsedResults']) > 0:
        return result['ParsedResults'][0]['ParsedText']
    else:
        return "لم يتم اكتشاف أي نص." if language == 'ara' else \
               "Kein Text erkannt." if language == 'ger' else \
               "No text detected."

def analyze_document(file, language='eng'):
    """
    Analyze the uploaded document and provide health recommendations.
    
    Parameters:
    file: The uploaded file object
    language: The language code ('eng', 'ara', or 'ger')
    """
    file_content = file.read()
    file_extension = file.filename.split('.')[-1].lower()

    if file_extension == 'pdf':
        text = extract_text_from_pdf(file_content)
    elif file_extension in ['png', 'jpg', 'jpeg']:
        text = extract_text_from_image(file_content, language)
    else:
        return "صيغة ملف غير مدعومة. يرجى رفع ملف PDF أو صورة." if language == 'ara' else \
               "Nicht unterstütztes Dateiformat. Bitte laden Sie eine PDF- oder Bilddatei hoch." if language == 'ger' else \
               "Unsupported file format. Please upload a PDF or image file."

    # Create a prompt for the chat model
    template = """
    You are a medical assistant AI. Analyze the following medical report and provide a detailed explanation and health recommendations based on the extracted data.
    
    Report language: {language}
    Report:
    {text}

    Provide the analysis in {language} in a structured format, including:
    - Summary of the report
    - Key findings
    - Health recommendations
    - Any warnings or alerts based on the results
    """

    prompt = ChatPromptTemplate.from_template(template)
    formatted_prompt = prompt.format_messages(text=text, language=language)

    # Get the response from the model
    response = model.invoke(formatted_prompt)
    return response.content