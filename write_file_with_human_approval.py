import operator
import os
from pathlib import Path
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langchain.tools import tool
from langchain_core.messages import AnyMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

load_dotenv(Path(__file__).parent / ".env", override=True)

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found. Add it to .env or your environment.")


# -----------------------------
# 1. Define the agent's state
# -----------------------------
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]


# -----------------------------
# 2. Define a sensitive tool with HITL interrupt
# -----------------------------
@tool
def write_to_file(filename: str, content: str) -> str:
    """Writes content to a file. Requires human approval before executing."""
    response = interrupt(
        {
            "action": "write_to_file",
            "filename": filename,
            "content": content,
            "message": "Approve writing this file?",
        }
    )
    print("\nReached here 1")
    if response.get("action") == "approve":
        final_filename = response.get("filename", filename)
        final_content = response.get("content", content)
        with open(final_filename, "w", encoding="utf-8") as f:
            f.write(final_content)
        print("\nReached here 2")
        return f"File '{final_filename}' written successfully."

    return "File write cancelled by user."


tools = [write_to_file]
tools_by_name = {t.name: t for t in tools}
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(tools)


# -----------------------------
# 3. Build the LangGraph workflow
# -----------------------------
def llm_node(state: AgentState):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


def tool_node(state: AgentState):
    results = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        results.append(
            ToolMessage(content=observation, tool_call_id=tool_call["id"])
        )
    return {"messages": results}


def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END


graph = StateGraph(AgentState)
graph.add_node("llm", llm_node)
graph.add_node("tools", tool_node)
graph.add_edge(START, "llm")
graph.add_conditional_edges("llm", should_continue, ["tools", END])
graph.add_edge("tools", "llm")

memory = MemorySaver()
app = graph.compile(checkpointer=memory)


def handle_interrupt(interrupts) -> dict:
    payload = interrupts[0].value
    print("\nAgent wants to write to a file:")
    print(f"  File: {payload.get('filename')}")
    print(f"  Content: {payload.get('content')}")

    approval = input("Approve? (y/n): ").strip().lower()
    if approval == "y":
        return {**payload, "action": "approve"}
    return {"action": "reject"}


# -----------------------------
# 4. Run the agent
# -----------------------------
if __name__ == "__main__":
    print("HITL Agent Started. Type 'exit' to quit.\n")
    thread_id = "session1"
    config = {"configurable": {"thread_id": thread_id}}

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        result = app.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
        )

        while result.get("__interrupt__"):
            resume_value = handle_interrupt(result["__interrupt__"])
            result = app.invoke(Command(resume=resume_value), config=config)

        last_message = result["messages"][-1]
        print("Agent:", last_message.content)
