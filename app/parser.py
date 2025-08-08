import requests
import tempfile
from pdfminer.high_level import extract_text
import docx
import logging

logger = logging.getLogger(__name__)

def parse_pdf(file_path: str) -> str:
    """Lighter PDF parsing alternative"""
    try:
        return extract_text(file_path)
    except Exception as e:
        logger.error(f"PDF Parse Error: {str(e)}")
        return ""

def parse_document(url: str) -> str:
    """Unified document parser with memory safety"""
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with tempfile.NamedTemporaryFile() as tmp:
                for chunk in r.iter_content(chunk_size=8192):
                    tmp.write(chunk)
                
                if url.endswith('.pdf'):
                    return parse_pdf(tmp.name)
                elif url.endswith('.docx'):
                    doc = docx.Document(tmp.name)
                    return "\n".join(p.text for p in doc.paragraphs)
                else:
                    raise ValueError("Unsupported format")
    except Exception as e:
        logger.error(f"Parse Error: {str(e)}")
        return ""
