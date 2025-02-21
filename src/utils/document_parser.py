from typing import Optional
import PyPDF2
import docx
import re


class DocumentParser:
    def __init__(self):
        self.supported_formats = ['.pdf', '.docx', '.txt']

    def read_file(self, file_path:str) -> Optional[str]:
        """Reads and extracts text from various document formats"""

        if file_path.endswith('.pdf'):
            return self._parse_pdf(file_path)
        elif file_path.endswith('.docx'):
            return self._parse_docx(file_path)
        elif file_path.endswith('.txt'):
            return self._parse_txt(file_path)
        else:
            raise ValueError(f'Unsupported file format: {file_path}. Supported formats: {self.supported_formats}')

    def _parse_pdf(self, file_path:str) -> str:
        """Parses a PDF file"""
        with open(file_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfFileReader(pdf_file)
            text = ''

            for page in pdf_reader.pages:
                text += page.extractText()

            return text

    def _parse_docx(self, file_path:str) -> str:
        """Parses a docx file"""
        doc = docx.Document(file_path)
        text = ''
        for paragraph in doc.paragraphs:
            text += paragraph.text + '\n'

        return text

    def _parse_txt(self, file_path:str) -> str:
        """Parses a txt file"""
        with open(file_path, 'r', encoding='utf-8') as txt_file:
            return txt_file.read()

    def clean_text(self, text:str) -> str:
        """Basic text cleaning"""
        text = ' '.join(text.split())

        text = re.sub(r'[^\w\s]', '', text)

        return text
