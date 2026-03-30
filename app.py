import os
import getpass
import sqlite3

# ── Environment setup ─────────────────────────────────────────────────────────

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("GROQ_API_KEY")

_set_env("LANGSMITH_API_KEY")
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "langchain-academy"

# ── Imports ───────────────────────────────────────────────────────────────────

from typing_extensions import Literal
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.checkpoint.sqlite import SqliteSaver

# ── DB + memory ───────────────────────────────────────────────────────────────

import os, subprocess

os.makedirs("state_db", exist_ok=True)
db_path = "state_db/example.db"
if not os.path.exists(db_path):
    subprocess.run([
        "wget", "-P", "state_db",
        "https://github.com/langchain-ai/langchain-academy/raw/main/module-2/state_db/example.db"
    ], check=True)

conn = sqlite3.connect(db_path, check_same_thread=False)
memory = SqliteSaver(conn)

# ── Model ─────────────────────────────────────────────────────────────────────

model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)

# ── State ─────────────────────────────────────────────────────────────────────

class State(MessagesState):
    summary: str

# ── Nodes ─────────────────────────────────────────────────────────────────────

def call_model(state: State):
    summary = state.get("summary", "")

    if summary:
        system_message = f"Summary of the conversation earlier: {summary}"
        messages = [SystemMessage(content=system_message)] + state["messages"]
    else:
        messages = state["messages"]

    response = model.invoke(messages)
    return {"messages": response}


def summarize_conversation(state: State):
    summary = state.get("summary", "")

    if summary:
        summary_message = (
            f"This is the summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above: "
        )
    else:
        summary_message = "Create a summary of the conversation above"

    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = model.invoke(messages)

    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    return {"summary": response.content, "messages": delete_messages}


def should_continue(state: State) -> Literal["summarize_conversation", END]:
    """Return the next node to execute."""
    messages = state["messages"]

    if len(messages) > 6:
        return "summarize_conversation"
    return END

# ── Graph ─────────────────────────────────────────────────────────────────────

workflow = StateGraph(State)
workflow.add_node("conversation", call_model)
workflow.add_node(summarize_conversation)

workflow.add_edge(START, "conversation")
workflow.add_conditional_edges("conversation", should_continue)
workflow.add_edge("summarize_conversation", END)

graph = workflow.compile(checkpointer=memory)

# ── Chat loop ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "1"}}

    print("Chatbot ready. Type 'quit' to exit.\n")

    while True:
        user_input = input("You: ")

        if user_input.lower() in ("quit", "exit", "q"):
            print("Ending conversation.")
            break

        input_message = HumanMessage(content=user_input)
        output = graph.invoke({"messages": [input_message]}, config)

        output["messages"][-1].pretty_print()

        summary = output.get("summary", "")
        if summary:
            print(f"\n[Active summary]: {summary}\n")