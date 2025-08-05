# app/parser.py
import requests
import mimetypes
import tempfile
import fitz  # PyMuPDF
import docx
from email import message_from_bytes
from bs4 import BeautifulSoup

def download_file(url: str) -> str:
    response = requests.get(url)
    response.raise_for_status()

    content_type = response.headers.get('content-type', '')
    extension = mimetypes.guess_extension(content_type.split(';')[0])
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=extension)
    temp_file.write(response.content)
    temp_file.close()
    return temp_file.name

def parse_pdf(file_path: str) -> str:
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def parse_docx(file_path: str) -> str:
    doc = docx.Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

def parse_email(file_path: str) -> str:
    with open(file_path, 'rb') as f:
        msg = message_from_bytes(f.read())
    payload = msg.get_payload()
    if isinstance(payload, list):
        for part in payload:
            if part.get_content_type() == 'text/html':
                soup = BeautifulSoup(part.get_payload(decode=True), 'html.parser')
                return soup.get_text()
            elif part.get_content_type() == 'text/plain':
                return part.get_payload(decode=True).decode()
    return str(payload)

def parse_document(url: str) -> str:
    file_path = download_file(url)
    if file_path.endswith('.pdf'):
        return parse_pdf(file_path)
    elif file_path.endswith('.docx'):
        return parse_docx(file_path)
    elif file_path.endswith('.eml'):
        return parse_email(file_path)
    else:
        raise ValueError("Unsupported document format")