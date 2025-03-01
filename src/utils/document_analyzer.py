import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from collections import Counter
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

try:
    nltk.data.find('tokenizers/punkt')
except:
    nltk.download('punkt')
try:
    nltk.find('corpora/stopwords')
except:
    nltk.download('stopwords')

class DocumentAnalyzer:
    def __init__(self, text, language='english'):
        self.text = text
        self.language = language
        self.sentences = sent_tokenize(text)
        self.words = word_tokenize(text.lower())

        try:
            self.stop_words = set(stopwords.words(language))
        except:
            self.stop_words = set(stopwords.words('english'))

        self.filtered_words = [word for word in self.words if word.isalnum() and word not in self.stop_words]


    def get_basic_stats(self):
        """Get basic text statistics"""
        word_count = len(self.words)
        sentence_count = len(self.sentences)
        avg_sentence_count = word_count / max(1, sentence_count)
        unique_words = len(set(self.filtered_words))

        return {
            "Document Length (words)": word_count,
            "Number of Sequences": sentence_count,
            "Average Sentence Lenght": round(avg_sentence_count, 2),
            "Unique Words": unique_words,
            "Lexical Diversity": round(unique_words / max(1, word_count), 3)
        }


    def get_keyword_distribution(self, top_n=20):
        """Get distribution of keywords"""
        word_freq = Counter(self.filtered_words)
        return word_freq.most_common(top_n)

    def get_readability_score(self ):
        """Calculate approximate readability score (Flesch-Kincaid)"""
        word_count = len(self.words)
        sentence_count = len(self.sentences)

        syllable_count = 0
        for word in self.words:
            syllable_count += self.count_syllables(word)

        if sentence_count > 0 and word_count > 0:
            score = 0.39 * (word_count / sentence_count) + 11.8 * (syllable_count/ word_count) - 15.59
            return round(score, 2)
        return 0

    def count_syllables(self, word):
        """Approximate syllable count"""
        word.lower()
        if len(word) <= 3:
            return 1

        word = re.sub(r'[^a-z]', '', word)
        vowels = "aeiouy"
        count = 0
        prev_is_vowel = False

        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_is_vowel:
                count += 1
            prev_is_vowel = is_vowel

        if word.endswith('e'):
            count += 1
        if count == 0:
            count = 1

        return count

    def generate_word_cloud(self):
        """Generate word cloud using wordcloud library"""
        text = ''.join(self.filtered_words)
        return WordCloud(
            width=800,
            height=400,
            background_color='black',
            colormap='viridis',
            max_words=100).generate(text)

def display_document_dashboard(text):
    """Display document analysis dashboard"""

    analyzer = DocumentAnalyzer(text)

    st.subheader("Document Statistics")
    stats = analyzer.get_basic_stats()

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Document Length", f"{stats['Document Length (words)']} words")
        st.metric("Number of Sequences", stats['Number of Sequences'])
        st.metric("Unique Words", stats['Unique Words'])

    with col2:
        st.metric("Average Sentence Lenght", f"{stats['Average Sentence Lenght']} words")
        st.metric("Lexical Diversity", f"{stats['Lexical Diversity'] * 100:.1f%} ")
        readability_score = analyzer.get_readability_score()
        st.metric("Readability Score", f"{readability_score} (approx. grade level)")

    st.subheader("Keyword Distribution")

    viz_tab1, viz_tab2 = st.tabs(["Word Cloud", "Bar Chart"])

    with viz_tab1:
        wordcloud = analyzer.generate_word_cloud()
        fix, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig=ax)

    with viz_tab2:
        keywords = analyzer.get_keyword_distribution()
        if keywords:
            df = pd.DataFrame(keywords, columns=['Word', 'Frequency'])
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(x='Frequency', y='Word', data=df, ax=ax)
            ax.set_title('Top 15 Keywords')
            st.pyplot(fig)
        else:
            st.info("Not enough data for keyword visualization")
def test_document_performance():
    """Test document performance section"""

    st.subheader("Test on Different Document Types")

    document_type = st.selectbox(
        "Select document to test",
        ["Technical Report", "Legal Contract", "News Article", "Scientific Paper"]
    )

    sample_texts = {
        "Technical Report": """
        System Performance Analysis Report
        """
    }