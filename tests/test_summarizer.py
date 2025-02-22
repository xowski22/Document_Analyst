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

    def test_chunk_text(self):
        long_text = "word" * 2000
        chunks = self.summarizer.chunk_text(long_text)
        self.assertTrue(len(chunks) > 1)
        self.assertTrue(all(len(chunk.split())) <= 1000 for chunk in chunks)

    def test_summarzation_with_chunks(self):
        long_text = "This is a long document. " * 100
        chunks = self.summarizer.chunk_text(long_text)

        for chunk in chunks:
            summary = self.summarizer.summarize(chunk)
            self.assertIsNotNone(summary)
            self.assertTrue(len(summary) < len(chunk))