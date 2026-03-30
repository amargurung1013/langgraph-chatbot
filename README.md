# LangGraph Chatbot

A conversational AI system built using LangGraph with memory management (trimming + summarization).

## Features

* Sliding window memory (token-based trimming)
* Optional conversation summarization
* Modular graph-based architecture

## Setup

```bash
git clone https://github.com/yourusername/langgraph-chatbot.git
cd langgraph-chatbot
pip install -r requirements.txt
```

## Run

```bash
python app.py
```

## Environment Variables

Create a `.env` file:

```
GROQ_API_KEY=your_key
LANGSMITH_API_KEY=your_key
```

## Tech Stack

* LangGraph
* LangChain
* OpenAI / Groq
