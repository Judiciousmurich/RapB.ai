from django.contrib import admin
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from api.views import DocumentUploadView, ChatView, ChatHistoryView

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/upload/', DocumentUploadView.as_view(), name='upload-document'),
    path('api/chat/', ChatView.as_view(), name='chat'),
    path('api/chat/history/', ChatHistoryView.as_view(), name='chat-history'),
    path('api/chat/history/<str:session_id>/',
         ChatHistoryView.as_view(), name='chat-session-history'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
