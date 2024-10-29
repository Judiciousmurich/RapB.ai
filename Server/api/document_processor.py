from transformers import pipeline
import torch
from typing import List, Dict, Any
import numpy as np
from langdetect import detect, LangDetectException


class DocumentProcessor:
    def __init__(self):
        # Initialize sentiment models for different languages
        self.device = 0 if torch.cuda.is_available() else -1

        # Initialize sentiment models without return_all_scores
        self.sentiment_models = {
            'en': pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=self.device
            ),
            'de': pipeline(
                "sentiment-analysis",
                model="oliverguhr/german-sentiment-bert",
                device=self.device
            )
        }

        # Default to English model if language not supported
        self.default_sentiment_model = self.sentiment_models['en']

        # Initialize embedding model
        self.embedding_model = pipeline(
            'feature-extraction',
            model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            device=self.device
        )

    def detect_language(self, text: str) -> str:
        """Detect the language of the input text."""
        try:
            sample = text[:1000]
            return detect(sample)
        except LangDetectException:
            return 'en'

    def analyze_sentiment(self, text: str, language: str = None) -> Dict[str, Any]:
        """Analyze sentiment of text using language-specific models."""
        try:
            if not language:
                language = self.detect_language(text)

            sentiment_model = self.sentiment_models.get(
                language, self.default_sentiment_model)
            result = sentiment_model(text[:512])[0]

            if language == 'de':
                score = result['score']
                label = result['label']
                if label == 'negative':
                    score = -score
            else:
                score = result['score']
                if result['label'] == 'NEGATIVE':
                    score = -score
                label = result['label']

            return {
                'score': score,
                'label': label,
                'language': language
            }
        except Exception as e:
            print(f"Error in sentiment analysis: {str(e)}")
            return {'score': 0.0, 'label': 'NEUTRAL', 'language': language or 'en'}

    def generate_embeddings(self, text: str) -> List[float]:
        """Generate embeddings for text using multilingual model."""
        try:
            embedding = self.embedding_model(text[:512])[0]
            return np.mean(embedding, axis=0).tolist()
        except Exception as e:
            print(f"Error generating embeddings: {str(e)}")
            return [0.0] * self.embedding_model.model.config.hidden_size

    def process_document(self, text: str) -> Dict[str, Any]:
        """Process document text with language detection."""
        try:
            language = self.detect_language(text)
            chunks = self.chunk_text(text)
            results = []

            for chunk in chunks:
                sentiment = self.analyze_sentiment(chunk, language)
                embedding = self.generate_embeddings(chunk)
                results.append({
                    'text': chunk,
                    'embedding': embedding,
                    'sentiment': sentiment['score']
                })

            sentiments = [r['sentiment'] for r in results]
            avg_sentiment = float(np.mean(sentiments)) if sentiments else 0.0

            return {
                'chunks': [r['text'] for r in results],
                'embeddings': [r['embedding'] for r in results],
                'sentiment': avg_sentiment,
                'detailed_sentiments': sentiments,
                'language': language,
                'chunk_count': len(chunks)
            }

        except Exception as e:
            print(f"Error processing document: {str(e)}")
            return {
                'chunks': [text],
                'embeddings': [self.generate_embeddings(text[:512])],
                'sentiment': 0.0,
                'detailed_sentiments': [0.0],
                'language': 'en',
                'chunk_count': 1
            }

    def chunk_text(self, text: str, max_length: int = 512) -> List[str]:
        """Split text into chunks based on simple sentence splitting."""
        sentences = []
        current = []
        lines = text.split('\n')
        text_with_breaks = ' [BREAK] '.join(lines)
        words = text_with_breaks.split()

        for word in words:
            current.append(word)
            if word.endswith(('.', '!', '?', '[BREAK]')) and len(current) >= 5:
                sentences.append(' '.join(current).replace(' [BREAK] ', '\n'))
                current = []

        if current:
            sentences.append(' '.join(current).replace(' [BREAK] ', '\n'))

        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence.split())
            if current_length + sentence_length > max_length:
                if current_chunk:
                    chunks.append('\n'.join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length

        if current_chunk:
            chunks.append('\n'.join(current_chunk))

        return chunks if chunks else [text]
