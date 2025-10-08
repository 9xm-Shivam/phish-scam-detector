# utils/text_extractor.py
import re
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader

def extract_text_from_pdf(path):
    text = ""
    with open(path, "rb") as f:
        reader = PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
    return text

def extract_text_from_html(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    return soup.get_text(separator=" ", strip=True)

def clean_text(text):
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()
