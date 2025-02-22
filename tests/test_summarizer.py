import unittest
from src.utils.document_parser import DocumentParser
from src.models.summarizer import DocumentSummarizer

class TestSummarizer(unittest.TestCase):
    def setUp(self):
        self.parser = DocumentParser()
        self.summarizer = DocumentSummarizer()

    def test_summarization(self):

        text = self.parser.read_file("sample.pdf")
        clean_text = self.parser.clean_text(text)
        chunks = self.summarizer.chunk_text(clean_text)

        summaries = []

        for chunk in chunks:
            summary = self.summarizer.summarize(chunk)
            summaries.append(summary)

        final_summary = " ".join(summaries)
        self.assertIsNotNone(final_summary)
        self.assertIsInstance(final_summary, str)