import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import START, END, StateGraph
from typing import TypedDict

load_dotenv(override=True)
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

class AgentState(TypedDict):
    question: str
    answer: str

def answer_question(state: AgentState) -> AgentState:
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    response = llm.invoke(state["question"])
    return {"answer": response.content}

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("answer_question", answer_question)
    graph.add_edge(START, "answer_question")
    graph.add_edge("answer_question", END)

    return graph


def main():
    graph = build_graph()
    app = graph.compile()
    result = app.invoke({"question": "What is the capital of France?", "answer": ""})
    print(result)

if __name__=="__main__":
    main()
