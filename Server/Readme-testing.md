# LYRIQ-AI API Testing Guide

## 1. Document Upload Test

- Method: POST
- URL: `http://localhost:8000/api/upload/`
- Body: form-data
  - Key: `file`
  - Value: Select `mixed_emotions_lyrics.txt`
- Expected Response:

```json
{
    "message": "Document processed successfully",
    "document_id": 1,
    "sentiment": [sentiment_score],
    "language": "en"
}
```

## 2. Start New Chat Session

- Method: POST
- URL: `http://localhost:8000/api/chat/`
- Headers:
  - Content-Type: application/json
- Body:

```json
{
  "message": "What emotions are expressed in these lyrics?",
  "document_id": 1
}
```

- Save the `session_id` from the response

## 3. Continue Chat with Follow-up Questions

- Method: POST
- URL: `http://localhost:8000/api/chat/`
- Headers:
  - Content-Type: application/json
- Body:

```json
{
  "session_id": "[session_id from previous response]",
  "message": "How does the mood change throughout the song?"
}
```

## 4. Try Different Questions

Test with these messages:

```json
{
  "session_id": "[your_session_id]",
  "message": "What's the main theme of the chorus?"
}
```

```json
{
  "session_id": "[your_session_id]",
  "message": "Is this generally a positive or negative song?"
}
```

```json
{
  "session_id": "[your_session_id]",
  "message": "What metaphors are used in these lyrics?"
}
```

## 5. View Chat History

- Method: GET
- URL: `http://localhost:8000/api/chat/history/[your_session_id]/`

## 6. View All Sessions

- Method: GET
- URL: `http://localhost:8000/api/chat/history/`

## 7. Test Error Cases

1. Invalid Document ID:

```json
{
  "message": "What's the meaning of these lyrics?",
  "document_id": 999
}
```

2. Invalid Session ID:

```json
{
  "session_id": "invalid-session-id",
  "message": "Hello"
}
```

3. Missing Message:

```json
{
  "session_id": "[your_session_id]"
}
```

## Expected Behaviors

1. Document Upload:

   - Should process and analyze sentiment
   - Should store document in system
   - Should return document_id

2. Chat:

   - Should maintain context within session
   - Should provide relevant responses based on lyrics
   - Should include sentiment analysis

3. History:
   - Should show all messages in chronological order
   - Should include sentiment scores
   - Should show document references
