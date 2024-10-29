import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Observable } from 'rxjs';

interface ChatRequest {
  message: string;
  document_id?: number;
  session_id?: string;
}

@Injectable({
  providedIn: 'root'
})
export class RapBotService {
  private apiUrl = 'http://localhost:8000/api';

  constructor(private http: HttpClient) { }

  // 1. Upload Document
  uploadDocument(file: File): Observable<any> {
    const formData = new FormData();
    formData.append('file', file);

    return this.http.post(`${this.apiUrl}/upload/`, formData);
  }

  // Updated startNewChat function
  startNewChat(message: string, documentId?: number): Observable<any> {
    const headers = new HttpHeaders({ 'Content-Type': 'application/json' });
    const body: any = documentId !== undefined ? { message } : { text: message };

    // If documentId is provided, use /chat endpoint, otherwise use /upload
    const endpoint = documentId !== undefined ? '/chat/' : '/upload/';

    if (documentId !== undefined) {
      body.document_id = documentId;
    }

    return this.http.post(`${this.apiUrl}${endpoint}`, body, { headers });
  }

  // 3. Continue Chat
  continueChat(message: string, sessionId: string,): Observable<any> {
    const headers = new HttpHeaders({ 'Content-Type': 'application/json' });
    const body: ChatRequest = {
      message,
      document_id: sessionId as unknown as number,
    };

    return this.http.post(`${this.apiUrl}/chat/`, body, { headers });
  }

  // 4. View Session History
  viewSessionHistory(sessionId: string): Observable<any> {
    return this.http.get(`${this.apiUrl}/chat/history/${sessionId}/`);
  }

  // 5. View All Sessions
  viewAllSessions(): Observable<any> {
    return this.http.get(`${this.apiUrl}/chat/history/`);
  }
}
