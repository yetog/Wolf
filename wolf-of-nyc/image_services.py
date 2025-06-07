
"""Image processing and OCR services for ScriptVoice."""

import pytesseract
from PIL import Image
from typing import Tuple


def extract_text_from_image(image) -> Tuple[str, str]:
    """Extract text from uploaded image using OCR."""
    if image is None:
        return "", '<div class="status-error">❌ Please upload an image</div>'
    
    try:
        # Use pytesseract to extract text
        text = pytesseract.image_to_string(Image.open(image))
        if text.strip():
            return text.strip(), '<div class="status-success">✅ Text extracted successfully</div>'
        else:
            return "", '<div class="status-error">❌ No text found in the image</div>'
    
    except Exception as e:
        return "", f'<div class="status-error">❌ Error extracting text: {str(e)}</div>'
