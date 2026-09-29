# Adaptive RAG Assistant (LangGraph + Flask)

## Run
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                   # then set OPENAI_API_KEY
python app.py                                          # http://127.0.0.1:5000
```

## Structure
```
app.py              Flask routes: /, /ask (JSON), /ask/stream (NDJSON), /health
config.py           All settings from environment variables
rag/state.py        GraphState
rag/nodes.py        route_question, retrieve_documents, grade_documents,
                    web_search, generate_answer, check_answer (+ decide_to_finish)
rag/graph.py        LangGraph wiring
rag/knowledge_base.py  In-memory vector store (sample docs or files in knowledge_base/)
templates/index.html   Dark chat UI with live execution path
```

## Flow
```
route_question -> retrieve_documents -> grade_documents -> generate_answer -> check_answer -> END
       \-> web_search -----------------------/  (also when no doc is relevant)
check_answer -> regenerate | web_search | END   (bounded by MAX_GENERATION_RETRIES + RECURSION_LIMIT)
```

## Notes
- Web search uses Tavily when `TAVILY_API_KEY` is set; otherwise a mock result is used.
- Put your own `.md`/`.txt` files in `knowledge_base/` and update `KB_TOPICS` so the router knows what they cover.
- API test: `curl -X POST localhost:5000/ask -H "Content-Type: application/json" -d '{"question":"What is LangGraph?"}'`
