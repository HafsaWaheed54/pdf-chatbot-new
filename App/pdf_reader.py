from PyPDF2 import PdfReader

def extract_text_from_pdf(pdf_file):
    text = ""
    pdf_reader = PdfReader(pdf_file)
    for page in pdf_reader.pages:
        extracted_text = page.extract_text()
        if extracted_text:  # Ensure the extracted text is not None
            text += extracted_text + "\n"
    return text if text else "No text extracted from PDF."