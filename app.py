"""Flask entry point for the Adaptive RAG Assistant."""
import json
import logging
import threading

from flask import Flask, Response, jsonify, render_template, request, stream_with_context

from config import ConfigError, get_settings
from rag.graph import build_graph

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("app")

app = Flask(__name__)
settings = get_settings()

_graph = None
_lock = threading.Lock()
MAX_QUESTION_CHARS = 1000


def get_graph():
    """Build the graph lazily so the app can start (and show errors) without a key."""
    global _graph
    with _lock:
        if _graph is None:
            _graph = build_graph(settings)
    return _graph


def _initial_state(question: str) -> dict:
    return {"question": question, "documents": [], "retries": 0,
            "web_searched": False, "needs_web_search": False, "steps": []}


def _sources(state: dict) -> list[dict]:
    return [{"source": d.metadata.get("source", "unknown"), "snippet": d.page_content[:180]}
            for d in state.get("documents", [])]


def _read_question():
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    if not question:
        return None, (jsonify(error="Please enter a question."), 400)
    if len(question) > MAX_QUESTION_CHARS:
        return None, (jsonify(error=f"Question is too long (max {MAX_QUESTION_CHARS} characters)."), 400)
    return question, None


def _final_payload(state: dict) -> dict:
    return {"answer": state.get("generation", ""),
            "path": [s["node"] for s in state.get("steps", [])],
            "steps": state.get("steps", []),
            "sources": _sources(state),
            "attempts": state.get("retries", 0),
            "grounded": state.get("grounded")}


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/health")
def health():
    return jsonify(status="ok")


@app.post("/ask")
def ask():
    """Run the whole workflow and return the final answer + execution path as JSON."""
    question, err = _read_question()
    if err:
        return err
    try:
        result = get_graph().invoke(_initial_state(question),
                                    config={"recursion_limit": settings.recursion_limit})
        return jsonify(_final_payload(result))
    except ConfigError as exc:
        return jsonify(error=str(exc)), 500
    except Exception:
        log.exception("Workflow failed")
        return jsonify(error="The workflow failed. Check the server logs."), 500


@app.post("/ask/stream")
def ask_stream():
    """Stream each node's step as newline-delimited JSON, then the final result."""
    question, err = _read_question()
    if err:
        return err

    def events():
        state = _initial_state(question)
        try:
            graph = get_graph()
            for update in graph.stream(state, config={"recursion_limit": settings.recursion_limit},
                                       stream_mode="updates"):
                for _node, delta in update.items():
                    state.update(delta or {})
                    if state["steps"]:
                        yield json.dumps({"type": "step", **state["steps"][-1]}) + "\n"
            yield json.dumps({"type": "final", **_final_payload(state)}) + "\n"
        except ConfigError as exc:
            yield json.dumps({"type": "error", "message": str(exc)}) + "\n"
        except Exception:
            log.exception("Streaming workflow failed")
            yield json.dumps({"type": "error", "message": "The workflow failed. Check the server logs."}) + "\n"

    return Response(stream_with_context(events()), mimetype="application/x-ndjson",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


if __name__ == "__main__":
    app.run(host=settings.host, port=settings.port, debug=settings.debug, threaded=True)
