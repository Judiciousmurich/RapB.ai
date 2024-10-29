# RapB.ai

## Research Goals
1.Data Chunking: Preserve context when segmenting large lyrics.

2.Quantization: Balance VRAM usage and precision.
 
3.Language Generalization: Test cross-language effectiveness (German to English).

4.Domain Adaptation: Explore generalizability of models trained on rap lyrics.


## Technology Stack

| Component   | Details                          |
|-------------|----------------------------------|
| Dataset     | German and English Rap Lyrics    |
| LLMs        | BERT, LLAMA2                     |
| Frontend    | Angular                          |
| Backend     | Django (REST API)                |
| Database    | Chroma        |
| Libraries   | LangChain, Pandas, Transformers  |


## Implementation
1.Data Collection: Gather German and English lyrics datasets.

2.Model Fine-tuning: Fine-tune LLMs on the dataset.

3.Database Integration: Connect a vector database to the LLM.

4.Testing: Evaluate the RAG architecture’s performance.

