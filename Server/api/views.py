jfrom typing import List, Dict, Any, Optional, Union
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser
from django.core.files.base import ContentFile
from django.db import transaction
from django.utils import timezone
from .models import UploadedDocument, ChatSession, ChatMessage
from .serializers import UploadedDocumentSerializer, ChatSessionSerializer
from .document_processor import DocumentProcessor
from .chroma_client import get_chroma_client, get_or_create_collection
import uuid
from transformers import pipeline, AutoTokenizer
import torch
import logging
from pathlib import Path
import io
import re
from functools import lru_cache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentUploadView(APIView):
    parser_classes = (MultiPartParser, FormParser, JSONParser)

    def post(self, request, *args, **kwargs):
        try:
            # Handle direct text input
            if 'text' in request.data:
                content = request.data['text']
                title = request.data.get('title', 'Direct Text Input')

                # Create document record
                document = UploadedDocument.objects.create(
                    title=title,
                    content=content
                )

            # Handle file upload
            elif 'file' in request.FILES:
                serializer = UploadedDocumentSerializer(data=request.data)
                if not serializer.is_valid():
                    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

                document = serializer.save()
                content = document.file.read()
                if isinstance(content, bytes):
                    content = content.decode('utf-8')

            else:
                return Response({
                    "error": "No content provided. Please provide either 'text' or 'file'."
                }, status=status.HTTP_400_BAD_REQUEST)

            # Process the content
            with transaction.atomic():
                processor = DocumentProcessor()
                result = processor.process_document(content)

                # Initialize ChromaDB
                chroma_client = get_chroma_client()
                collection = get_or_create_collection(chroma_client)

                # Store chunks with embeddings
                for i, (chunk, embedding) in enumerate(zip(result['chunks'], result['embeddings'])):
                    collection.add(
                        embeddings=[embedding],
                        documents=[chunk],
                        ids=[f"{document.id}-chunk-{i}"],
                        metadatas=[{
                            "document_id": str(document.id),
                            "chunk_index": i,
                            "sentiment": result['detailed_sentiments'][i]
                        }]
                    )

                # Update document
                document.content = content
                document.processed = True
                document.language = result.get('language', 'en')
                document.average_sentiment = result['sentiment']
                document.save()

                return Response({
                    "message": "Content processed successfully",
                    "document_id": document.id,
                    "sentiment": result['sentiment'],
                    "language": result.get('language', 'en'),
                    "chunk_count": len(result['chunks'])
                }, status=status.HTTP_201_CREATED)

        except Exception as e:
            logger.error(f"Error processing content: {str(e)}", exc_info=True)
            if 'document' in locals():
                document.delete()  # Cleanup if document was created
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class ChatView(APIView):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = 0 if torch.cuda.is_available() else -1

        # Initialize multilingual model
        try:
            self.generator = pipeline(
                "text2text-generation",
                model="google/flan-t5-large",
                device=0 if torch.cuda.is_available() else -1,
                model_kwargs={
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "top_k": 50,
                    "repetition_penalty": 1.2,
                    "length_penalty": 1.0,
                    "early_stopping": True
                }
            )
            logger.info("Successfully initialized MT5 model")
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            raise

        # Initialize response cache
        self._response_cache = {}

        # Define supported languages
        self.supported_languages = {'en', 'de'}

    def get_analysis_prompt(self, question_type: str, language: str = 'en') -> str:
        """Generate language-specific analysis prompts"""

        prompts = {
            'en': {
                'sentiment': """Using only specific evidence from these lyrics, analyze their emotional content by:
1. Identifying explicit emotional words and phrases (quote them)
2. Noting any emotional changes throughout the text
3. Discussing the intensity of emotions shown
4. Connecting emotions to specific quoted lines

Remember: Only discuss emotions directly expressed in the lyrics.""",

                'theme': """Using only specific evidence from these lyrics, analyze their themes by:
1. Identifying main themes with direct quotes
2. Pointing out specific metaphors or symbols
3. Showing how themes develop using line references
4. Connecting themes to explicit evidence

Remember: Only discuss themes directly present in the lyrics.""",

                'structure': """Using only specific evidence from these lyrics, analyze their structure by:
1. Mapping out the exact organization (verse/chorus/etc.)
2. Identifying specific patterns or repetitions
3. Showing how structure affects meaning
4. Noting unique structural elements

Remember: Only discuss structural elements directly present in the lyrics.""",

                'general': """Using only specific evidence from these lyrics, provide analysis by:
1. Identifying main ideas with direct quotes
2. Noting specific techniques used
3. Discussing clear patterns or elements
4. Supporting all points with exact quotes

Remember: Only discuss what is explicitly present in the lyrics."""
            },

            'de': {
                'sentiment': """Analysieren Sie die emotionalen Aspekte dieser Lyrics, basierend nur auf dem Text:
1. Identifizieren Sie konkrete emotionale Wörter und Phrasen (mit Zitaten)
2. Beschreiben Sie die emotionale Entwicklung im Text
3. Diskutieren Sie die Intensität der gezeigten Gefühle
4. Belegen Sie Ihre Analyse mit spezifischen Textstellen

Wichtig: Beziehen Sie sich nur auf explizit im Text vorhandene Emotionen.""",

                'theme': """Analysieren Sie die Hauptthemen dieser Lyrics, basierend nur auf dem Text:
1. Identifizieren Sie die zentralen Themen mit direkten Zitaten
2. Zeigen Sie verwendete Metaphern oder Symbole
3. Beschreiben Sie die Entwicklung der Themen
4. Belegen Sie alle Punkte mit Textzitaten

Wichtig: Beziehen Sie sich nur auf explizit im Text vorhandene Themen.""",

                'structure': """Analysieren Sie den Aufbau dieser Lyrics:
1. Beschreiben Sie die genaue Organisation (Strophe/Refrain)
2. Identifizieren Sie Muster und Wiederholungen
3. Zeigen Sie die Beziehung zwischen Struktur und Bedeutung
4. Notieren Sie besondere strukturelle Elemente

Wichtig: Beziehen Sie sich nur auf die tatsächliche Textstruktur.""",

                'general': """Analysieren Sie diese Lyrics objektiv:
1. Identifizieren Sie Hauptaussagen mit Zitaten
2. Beschreiben Sie verwendete Techniken
3. Zeigen Sie klare Muster
4. Belegen Sie alle Punkte mit Textstellen

Wichtig: Bleiben Sie bei den expliziten Textinhalten."""
            }
        }

        language_prompts = prompts.get(language, prompts['en'])
        return language_prompts.get(question_type, language_prompts['general'])

    def determine_question_type(self, question: str) -> str:
        """Determine question type with multilingual support"""
        question = question.lower()

        patterns = {
            'sentiment': {
                'en': r'\b(emotion|feel|sentiment|mood|tone|attitude|express)\b',
                'de': r'\b(gefühl|emotion|stimmung|ausdruck|ton|haltung)\b'
            },
            'theme': {
                'en': r'\b(theme|meaning|message|about|discuss|topic|subject)\b',
                'de': r'\b(thema|bedeutung|botschaft|über|diskutieren|inhalt)\b'
            },
            'structure': {
                'en': r'\b(structure|pattern|organize|form|arrangement|layout)\b',
                'de': r'\b(struktur|muster|aufbau|form|anordnung|gliederung)\b'
            }
        }

        # Check patterns in both languages
        for qtype, lang_patterns in patterns.items():
            if any(re.search(pattern, question) for pattern in lang_patterns.values()):
                return qtype

        return 'general'

    def format_context(self, chunks: List[str], metadata: List[Dict] = None, language: str = 'en') -> str:
        """Format context with language-specific section names"""
        if not chunks:
            return ""

        def detect_section_type(text: str, index: int, total: int, language: str) -> str:
            text_lower = text.lower()

            if language == 'de':
                if re.search(r'\b(refrain|chorus)\b', text_lower):
                    return "Refrain"
                elif re.search(r'\b(bridge)\b', text_lower):
                    return "Bridge"
                elif index == 0 and re.search(r'\b(intro)\b', text_lower):
                    return "Intro"
                elif index == total - 1 and re.search(r'\b(outro)\b', text_lower):
                    return "Outro"
                return f"Strophe {index + 1}"
            else:
                if re.search(r'\b(chorus|refrain)\b', text_lower):
                    return "Chorus"
                elif re.search(r'\b(bridge)\b', text_lower):
                    return "Bridge"
                elif index == 0 and re.search(r'\b(intro)\b', text_lower):
                    return "Intro"
                elif index == total - 1 and re.search(r'\b(outro)\b', text_lower):
                    return "Outro"
                return f"Verse {index + 1}"

        formatted_sections = []
        total_chunks = len(chunks)

        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue

            meta = metadata[i] if metadata else {}
            sentiment_score = meta.get('sentiment', 0.0)

            section_type = detect_section_type(
                chunk, i, total_chunks, language)
            section_header = f"{section_type}"

            if sentiment_score:
                sentiment_label = ("Positiv" if sentiment_score > 0 else "Negativ") if language == 'de' else \
                    ("Positive" if sentiment_score > 0 else "Negative")
                sentiment_strength = ("Stark" if abs(sentiment_score) > 0.7 else "Moderat") if language == 'de' else \
                    ("Strong" if abs(sentiment_score) > 0.7 else "Moderate")
                section_header += f" [{sentiment_strength} {sentiment_label}]"

            formatted_sections.append(f"{section_header}:\n{chunk.strip()}")

        return "\n\n".join(formatted_sections)

    @torch.no_grad()
    def generate_response(self, context: str, question: str, language: str = 'en') -> str:
        """Generate language-appropriate response"""
        try:
            cache_key = f"{hash(context)}-{hash(question)}-{language}"
            if cache_key in self._response_cache:
                return self._response_cache[cache_key]

            question_type = self.determine_question_type(question)
            analysis_prompt = self.get_analysis_prompt(question_type, language)

            # Construct language-specific prompt
            if language == 'de':
                prompt = f"""Anweisung: Analysieren Sie die folgenden Lyrics ausschließlich basierend auf dem gegebenen Text.
Ihre Analyse muss:
1. Direkte Zitate aus dem Text verwenden
2. Keine Spekulationen enthalten
3. Sich nur auf explizit vorhandene Elemente beziehen
4. Die gestellte Frage konkret beantworten

Lyrics:
{context}

Frage: {question}

Analyserichtlinien:
{analysis_prompt}

Antwort:"""
            else:
                prompt = f"""Instructions: Provide a specific analysis based only on the given lyrics.
Your analysis must:
1. Quote directly from the lyrics to support every point
2. Avoid speculation or external interpretation
3. Focus only on what is explicitly present in the text
4. Address the specific question asked

Lyrics:
{context}

Question: {question}

Analysis Guidelines:
{analysis_prompt}

Response:"""

            # Generate response with error handling
            try:
                response = self.generator(
                    prompt,
                    max_length=750,
                    min_length=150,
                    num_return_sequences=1,
                    do_sample=True,
                    no_repeat_ngram_size=4
                )

                generated_text = response[0]['generated_text'].strip()

                if not generated_text or generated_text.isspace():
                    return self._get_fallback_response(language)

                processed_response = self._post_process_response(
                    generated_text, language)
                self._response_cache[cache_key] = processed_response

                return processed_response

            except Exception as e:
                logger.error(f"Error in response generation: {str(e)}")
                return self._get_fallback_response(language)

        except Exception as e:
            logger.error(f"Error in generate_response: {str(e)}")
            return self._get_fallback_response(language)

    def _get_fallback_response(self, language: str) -> str:
        """Provide language-appropriate fallback response"""
        if language == 'de':
            return "Entschuldigung, ich konnte keine aussagekräftige Analyse generieren. Bitte formulieren Sie Ihre Frage anders."
        return "I apologize, but I couldn't generate a meaningful analysis. Please try rephrasing your question."

    def _post_process_response(self, text: str, language: str) -> str:
        """Clean and structure the response with language awareness"""
        # Remove repeated sentences
        sentences = text.split('. ')
        unique_sentences = []
        seen = set()

        for sentence in sentences:
            normalized = re.sub(r'\s+', ' ', sentence.lower().strip())
            if normalized not in seen and len(normalized) > 10:
                seen.add(normalized)
                unique_sentences.append(sentence)

        # Join sentences and format paragraphs
        cleaned_text = '. '.join(unique_sentences)
        cleaned_text = re.sub(r'([.!?])\s*(?=[A-Z])', r'\1\n\n', cleaned_text)

        # Remove empty lines and normalize spacing
        cleaned_text = re.sub(r'\n\s*\n', '\n\n', cleaned_text)

        return cleaned_text.strip()

    @transaction.atomic
    def post(self, request, *args, **kwargs):
        """Handle chat interactions with full error handling"""
        try:
            session_id = request.data.get('session_id')
            message = request.data.get('message')
            document_id = request.data.get('document_id')

            if not message:
                return Response({"error": "Message is required"}, status=status.HTTP_400_BAD_REQUEST)

            # Session management
            if not session_id:
                session_id = str(uuid.uuid4())
                session = ChatSession.objects.create(session_id=session_id)
            else:
                try:
                    session = ChatSession.objects.get(session_id=session_id)
                except ChatSession.DoesNotExist:
                    return Response({"error": "Invalid session ID"}, status=status.HTTP_404_NOT_FOUND)

            # Document handling
            if document_id:
                try:
                    document = UploadedDocument.objects.get(id=document_id)
                    session.current_document = document
                    session.save()
                except UploadedDocument.DoesNotExist:
                    return Response({"error": "Document not found"}, status=status.HTTP_404_NOT_FOUND)
            else:
                if not session.current_document:
                    return Response({"error": "Document ID is required for querying."}, status=status.HTTP_400_BAD_REQUEST)
                document = session.current_document

            # Get document language
            document_language = document.language if document.language in self.supported_languages else 'en'

            # Process user message
            processor = DocumentProcessor()
            sentiment_result = processor.analyze_sentiment(
                message, document_language)

            # Save user message
            user_message = ChatMessage.objects.create(
                session=session,
                content=message,
                is_user=True,
                sentiment_score=sentiment_result['score']
            )

            # Get context from ChromaDB
            query_embedding = processor.generate_embeddings(message)
            chroma_client = get_chroma_client()
            collection = get_or_create_collection(chroma_client)

            # Query with document filter
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=5,
                where={"document_id": str(document.id)}
            )

            if not results['documents'][0]:
                return Response({
                    "session_id": session_id,
                    "response": self._get_fallback_response(document_language),
                    "user_sentiment": sentiment_result,
                    "response_sentiment": 0.0
                })

            # Generate response
            context = self.format_context(
                results['documents'][0],
                results['metadatas'][0],
                document_language
            )
            response = self.generate_response(
                context, message, document_language)

            # Analyze response sentiment
            response_sentiment = processor.analyze_sentiment(
                response, document_language)

            # Save bot response
            bot_message = ChatMessage.objects.create(
                session=session,
                content=response,
                is_user=False,
                sentiment_score=response_sentiment['score'],
                relevant_document=document
            )

            # Update session
            # Update session
            session.last_interaction = timezone.now()
            session.save()

            return Response({
                "session_id": session_id,
                "response": response,
                "user_sentiment": sentiment_result,
                "response_sentiment": response_sentiment['score'],
                "document_id": document.id,
                "language": document_language,
                "metadata": {
                    "question_type": self.determine_question_type(message),
                    "context_chunks": len(results['documents'][0]),
                    "document_title": document.title
                }
            })

        except Exception as e:
            logger.error(f"Error in chat processing: {str(e)}", exc_info=True)
            return Response({
                "error": str(e),
                "detail": "An error occurred while processing your request."
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    def _validate_language(self, language: str) -> str:
        """Validate and return supported language code"""
        if language not in self.supported_languages:
            logger.warning(
                f"Unsupported language: {language}, falling back to English")
            return 'en'
        return language

    def clear_cache(self):
        """Clear the response cache"""
        self._response_cache.clear()
        logger.info("Response cache cleared")

    def __del__(self):
        """Cleanup when the view is destroyed"""
        try:
            self.clear_cache()
            if hasattr(self, 'generator'):
                del self.generator
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"Error in cleanup: {str(e)}")


class ChatHistoryView(APIView):
    """Handle chat history operations"""

    def get(self, request, session_id=None):
        """Get chat history for a session or list all sessions"""
        try:
            if session_id:
                session = ChatSession.objects.get(session_id=session_id)

                # Get document info
                document_info = None
                if session.current_document:
                    document_info = {
                        'id': session.current_document.id,
                        'title': session.current_document.title or session.current_document.file.name,
                        'uploaded_at': session.current_document.uploaded_at,
                        'language': session.current_document.language,
                        'average_sentiment': session.current_document.average_sentiment
                    }

                # Get messages with pagination
                messages = ChatMessage.objects.filter(
                    session=session).order_by('timestamp')
                messages_data = [{
                    'content': msg.content,
                    'is_user': msg.is_user,
                    'sentiment_score': msg.sentiment_score,
                    'timestamp': msg.timestamp,
                    'relevant_document_id': msg.relevant_document.id if msg.relevant_document else None
                } for msg in messages]

                return Response({
                    'session_id': session.session_id,
                    'created_at': session.created_at,
                    'last_interaction': session.last_interaction,
                    'current_document': document_info,
                    'messages': messages_data,
                    'message_count': len(messages_data)
                })

            else:
                # Return list of all chat sessions with pagination
                page = int(request.query_params.get('page', 1))
                page_size = int(request.query_params.get('page_size', 10))
                start = (page - 1) * page_size
                end = start + page_size

                sessions = ChatSession.objects.all().order_by(
                    '-last_interaction')[start:end]

                sessions_data = [{
                    'session_id': session.session_id,
                    'created_at': session.created_at,
                    'last_interaction': session.last_interaction,
                    'document': {
                        'title': session.current_document.title if session.current_document else None,
                        'id': session.current_document.id if session.current_document else None,
                        'language': session.current_document.language if session.current_document else None
                    },
                    'message_count': session.messages.count(),
                    'last_message': session.messages.order_by('-timestamp').first().content if session.messages.exists() else None
                } for session in sessions]

                total_sessions = ChatSession.objects.count()

                return Response({
                    'sessions': sessions_data,
                    'total': total_sessions,
                    'page': page,
                    'page_size': page_size,
                    'total_pages': (total_sessions + page_size - 1) // page_size
                })

        except ChatSession.DoesNotExist:
            return Response(
                {"error": "Session not found"},
                status=status.HTTP_404_NOT_FOUND
            )
        except Exception as e:
            logger.error(
                f"Error retrieving chat history: {str(e)}", exc_info=True)
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    def delete(self, request, session_id):
        """Delete a chat session and its associated messages"""
        try:
            with transaction.atomic():
                session = ChatSession.objects.get(session_id=session_id)
                # Delete all associated messages first
                ChatMessage.objects.filter(session=session).delete()
                # Delete the session
                session.delete()
                return Response(
                    {"message": "Chat session and associated messages deleted successfully"},
                    status=status.HTTP_200_OK
                )
        except ChatSession.DoesNotExist:
            return Response(
                {"error": "Session not found"},
                status=status.HTTP_404_NOT_FOUND
            )
        except Exception as e:
            logger.error(
                f"Error deleting chat session: {str(e)}", exc_info=True)
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    def patch(self, request, session_id):
        """Update chat session properties"""
        try:
            session = ChatSession.objects.get(session_id=session_id)

            # Update allowed fields
            if 'title' in request.data:
                session.title = request.data['title']

            if 'document_id' in request.data:
                try:
                    document = UploadedDocument.objects.get(
                        id=request.data['document_id'])
                    session.current_document = document
                except UploadedDocument.DoesNotExist:
                    return Response({"error": "Document not found"}, status=status.HTTP_404_NOT_FOUND)

            session.save()

            return Response({
                "message": "Session updated successfully",
                "session_id": session.session_id,
                "title": session.title,
                "document_id": session.current_document.id if session.current_document else None
            })

        except ChatSession.DoesNotExist:
            return Response({"error": "Session not found"}, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            logger.error(
                f"Error updating chat session: {str(e)}", exc_info=True)
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
