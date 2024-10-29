#!/usr/bin/env python
from setuptools import setup, find_packages

setup(
    name="lyriq-ai",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        'django',
        'djangorestframework',
        'django-cors-headers',
        'chromadb',
        'langchain',
        'torch',
        'transformers',
        'sentencepiece',
        'protobuf',
        'langdetect',
        'numpy',
        'scikit-learn',
        'pandas',
        'python-dotenv',
        'tqdm',
        'accelerate',
        'peft',
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="RAP-B: A RAG-based chatbot for analyzing rap lyrics",
    keywords="nlp, rag, chatbot, lyrics, sentiment-analysis",
)
