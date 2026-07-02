# filename: test_llm_workflow.py
import pytest
from llm_workflow import build_graph

@pytest.mark.parametrize("question, expected_keyword", [
    ("What is the capital of France?", "Paris"),
    ("Who wrote Hamlet?", "Shakespeare"),
])
def test_llm_responses(question, expected_keyword):
    app = build_graph().compile()
    result = app.invoke({"question": question, "answer": ""})
    
    # Basic validation: check if expected keyword is in the answer
    assert expected_keyword.lower() in result["answer"].lower(), \
        f"Expected '{expected_keyword}' in answer, got: {result['answer']}"
