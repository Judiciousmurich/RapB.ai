from django.contrib import admin
from .models import UploadedDocument, ChatSession, ChatMessage


@admin.register(UploadedDocument)
class UploadedDocumentAdmin(admin.ModelAdmin):
    list_display = ('title', 'uploaded_at', 'processed',
                    'language', 'average_sentiment')
    list_filter = ('processed', 'language', 'uploaded_at')
    search_fields = ('title', 'content')


@admin.register(ChatSession)
class ChatSessionAdmin(admin.ModelAdmin):
    list_display = ('session_id', 'created_at', 'last_interaction')
    list_filter = ('created_at', 'last_interaction')
    search_fields = ('session_id',)


@admin.register(ChatMessage)
class ChatMessageAdmin(admin.ModelAdmin):
    list_display = ('session', 'is_user', 'sentiment_score', 'timestamp')
    list_filter = ('is_user', 'timestamp')
    search_fields = ('content',)
