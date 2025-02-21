import unittest

from src.utils.document_parser import DocumentParser

class TestDocumentParser(unittest.TestCase):
    def setUp(self):
        self.parser = DocumentParser()

    def test_txt_parsing(self):
        text = self.parser.read_file("sample.txt")
        self.assertIsNotNone(text)
        self.assertIsInstance(text, str)