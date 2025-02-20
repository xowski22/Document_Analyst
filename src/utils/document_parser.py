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
        pass

    def _parse_docx(self, file_path:str) -> str:
        """Parses a docx file"""
        pass

    def _parse_txt(self, file_path:str) -> str:
        """Parses a txt file"""
        pass

    def clean_text(self, text:str) -> str:
        """Basic text cleaning"""
        pass

