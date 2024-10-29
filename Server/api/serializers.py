from rest_framework import serializers
from .models import UploadedDocument, ChatSession, ChatMessage


class UploadedDocumentSerializer(serializers.ModelSerializer):
    class Meta:
        model = UploadedDocument
        fields = ['id', 'file', 'uploaded_at',
                  'processed', 'title', 'average_sentiment']
        read_only_fields = ['uploaded_at', 'processed', 'average_sentiment']


class ChatMessageSerializer(serializers.ModelSerializer):
    class Meta:
        model = ChatMessage
        fields = ['content', 'is_user', 'sentiment_score', 'timestamp']


class ChatSessionSerializer(serializers.ModelSerializer):
    messages = ChatMessageSerializer(many=True, read_only=True)

    class Meta:
        model = ChatSession
        fields = ['session_id', 'created_at', 'last_interaction', 'messages']
