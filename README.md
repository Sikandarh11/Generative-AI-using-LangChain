# Generative AI using LangChain

A structured, hands-on learning repository that walks through the core building blocks of LangChain — from prompt engineering and output parsing all the way through to retrieval-augmented generation (RAG) and agentic AI systems.

---

## Table of Contents

1. [Overview](#overview)
2. [Repository Structure](#repository-structure)
3. [Learning Roadmap](#learning-roadmap)
4. [Prerequisites](#prerequisites)
5. [Setup Instructions](#setup-instructions)
6. [Environment Variables & API Keys](#environment-variables--api-keys)
7. [Git Submodule](#git-submodule)
8. [Running the Examples](#running-the-examples)
   - [Python Scripts](#python-scripts)
   - [Jupyter Notebooks](#jupyter-notebooks)
9. [Troubleshooting](#troubleshooting)
10. [Contributing](#contributing)
11. [License](#license)

---

## Overview

This repository is a progressive learning path for building Generative AI applications with [LangChain](https://python.langchain.com/). Each numbered folder focuses on a distinct concept, moving from the basics of LLM prompting to full agentic workflows. By working through the modules in order you will learn how to:

- Craft and templatize prompts for various LLM providers.
- Enforce structured (typed) outputs from language models.
- Parse and transform raw LLM responses with output parsers.
- Build sequential and parallel processing pipelines with LangChain chains (LCEL / `Runnable`).
- Load, split, embed, and retrieve documents for RAG applications.
- Persist vector embeddings in a local vector store.
- Build a real-world YouTube chat application.
- Use LangChain's built-in tools and agents.
- Integrate an external agentic AI project via a Git submodule.

---

## Repository Structure

```
Generative-AI-using-LangChain/
├── 1. prompts/                          # Prompt templates, chat history, Streamlit UI
├── 2. structured-output/                # TypedDict-based structured LLM output
├── 3. output parser/                    # String output parser
├── 4. chains/                           # LCEL chains (simple, runnable, parallel)
├── 5. document_loader/                  # Document loading (PDF, CSV, TXT)
├── 6. Text Splitters/                   # Chunking strategies for long documents
├── 7. vector store/                     # Embedding + vector store persistence
├── 8. retrivers/                        # LangChain retrievers for RAG
├── 9. Youtube chat project/             # End-to-end YouTube transcript chat app
├── 10. Built_in_tools/                  # LangChain built-in tool calling
├── 11-agentic_ai_project_Git_Submodule/ # Git submodule → AI-Agent-Using-LangChain
├── .gitignore
├── .gitmodules
└── README.md
```

### Module Descriptions

| # | Folder | Description |
|---|--------|-------------|
| 1 | `1. prompts` | Covers `PromptTemplate`, `ChatPromptTemplate`, `MessagesPlaceholder`, message types, temperature tuning, a prompt generator utility, and a Streamlit-based prompt UI. |
| 2 | `2. structured-output` | Demonstrates how to get type-safe structured output from an LLM using Python `TypedDict` and JSON schema. |
| 3 | `3. output parser` | Shows the `StrOutputParser` for cleaning raw LLM string responses. |
| 4 | `4. chains` | Builds LangChain pipelines with LCEL — simple chains, runnable sequences, and manual runnable composition. Includes a practical Bollywood/Hindi film demo. |
| 5 | `5. document_loader` | Loads real-world documents: a PDF (`dl-curriculum.pdf`), a plain-text file (`cricket.txt`), a CSV (`Social_Network_Ads.csv`), and a PDF book. |
| 6 | `6. Text Splitters` | Explores character and recursive text splitters to chunk documents before embedding. |
| 7 | `7. vector store` | Creates embeddings, stores them in a local vector store, and demonstrates similarity search. |
| 8 | `8. retrivers` | Introduces LangChain retrievers and shows how to wire them into a RAG chain. |
| 9 | `9. Youtube chat project` | A complete project: fetches a YouTube video transcript, embeds it, and answers questions about the video using RAG. |
| 10 | `10. Built_in_tools` | Explores LangChain's built-in tools (e.g., search, calculator) and tool-calling patterns. |
| 11 | `11-agentic_ai_project_Git_Submodule` | Git submodule pointing to [AI-Agent-Using-LangChain](https://github.com/Sikandarh11/AI-Agent-Using-LangChain.git) — a full agentic AI project. |

---

## Learning Roadmap

Work through the modules in numerical order for the most coherent learning experience:

```
Prompts (1)
  └─► Structured Output (2)
        └─► Output Parser (3)
              └─► Chains / LCEL (4)
                    └─► Document Loader (5)
                          └─► Text Splitters (6)
                                └─► Vector Store (7)
                                      └─► Retrievers (8)
                                            └─► YouTube Chat Project (9)
                                                  └─► Built-in Tools (10)
                                                        └─► Agentic AI (11)
```

---

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| **Python 3.10+** | Recommended; 3.11 or 3.12 also work. |
| **pip** | Comes with Python. |
| **Virtual environment** | `venv` (built-in) or `conda`. |
| **Jupyter** | `pip install jupyter` or use VS Code's Jupyter extension. |
| **API key(s)** | At least one LLM provider key — see [Environment Variables](#environment-variables--api-keys). |

> **Note:** Each module may have its own dependency requirements (e.g., `langchain-openai`, `langchain-google-genai`, `chromadb`, `youtube-transcript-api`). Check the import statements at the top of each notebook/script and install what is needed.

---

## Setup Instructions

### 1. Clone the repository (including the submodule)

```bash
git clone --recurse-submodules https://github.com/Sikandarh11/Generative-AI-using-LangChain.git
cd Generative-AI-using-LangChain
```

If you already cloned without `--recurse-submodules`, initialize the submodule manually:

```bash
git submodule init
git submodule update
```

### 2. Create and activate a virtual environment

```bash
# Create
python -m venv myvenv

# Activate (macOS / Linux)
source myvenv/bin/activate

# Activate (Windows PowerShell)
myvenv\Scripts\Activate.ps1
```

### 3. Install core dependencies

There is no single top-level `requirements.txt` because each module can have different dependencies. Install what you need as you work through each folder, for example:

```bash
# Core LangChain
pip install langchain langchain-core langchain-community

# LLM provider integrations (install the one(s) you use)
pip install langchain-openai          # OpenAI / Azure OpenAI
pip install langchain-google-genai    # Google Gemini
pip install langchain-anthropic       # Anthropic Claude

# Jupyter
pip install jupyter notebook

# Vector store & embeddings
pip install chromadb

# Document loaders
pip install pypdf                     # PDF loading
pip install youtube-transcript-api pytube  # YouTube chat project

# Streamlit (module 1 UI)
pip install streamlit
```

---

## Environment Variables & API Keys

The notebooks and scripts read credentials from environment variables (never hard-coded). Create a `.env` file in the repository root (it is already listed in `.gitignore` so it will not be committed):

```bash
# .env  — do NOT commit this file
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=AIza...
ANTHROPIC_API_KEY=sk-ant-...
LANGCHAIN_API_KEY=ls__...          # Optional: LangSmith tracing
LANGCHAIN_TRACING_V2=true          # Optional: enable LangSmith tracing
LANGCHAIN_PROJECT=my-project       # Optional: LangSmith project name
```

Load the `.env` file in your code with [`python-dotenv`](https://pypi.org/project/python-dotenv/):

```python
from dotenv import load_dotenv
load_dotenv()
```

Or export variables directly in your shell before running any script:

```bash
export OPENAI_API_KEY="sk-..."
```

> **Security reminder:** Never commit API keys to version control. The `.gitignore` already excludes `.env` and `env.*` files.

---

## Git Submodule

Module 11 is a Git **submodule** that references a separate repository:

- **Submodule path:** `11-agentic_ai_project_Git_Submodule`
- **Remote URL:** `https://github.com/Sikandarh11/AI-Agent-Using-LangChain.git`
- **Defined in:** `.gitmodules`

### Useful submodule commands

```bash
# Clone parent repo and automatically initialize + populate the submodule
git clone --recurse-submodules https://github.com/Sikandarh11/Generative-AI-using-LangChain.git

# Initialize + populate an already-cloned repo (run once after a plain clone)
git submodule init
git submodule update

# Short-hand for the two commands above
git submodule update --init

# Pull latest commits in the parent repo AND update all submodules
git pull --recurse-submodules

# Advance the submodule to the latest commit on its remote default branch
git submodule update --remote 11-agentic_ai_project_Git_Submodule
```

Navigate into the submodule directory to explore it independently:

```bash
cd 11-agentic_ai_project_Git_Submodule
# The submodule has its own README, requirements, and Git history
```

---

## Running the Examples

### Python Scripts

Module 1 contains standalone Python scripts. Run them from the module directory after activating your virtual environment:

```bash
cd "1. prompts"
python prompt_template.py
python chatbot.py
python prompt_ui.py   # Streamlit app — opens in the browser
```

For the Streamlit UI:

```bash
streamlit run "1. prompts/prompt_ui.py"
```

### Jupyter Notebooks

All other modules use Jupyter notebooks (`.ipynb`). Launch the Jupyter server from the repository root:

```bash
jupyter notebook
# or
jupyter lab
```

Then navigate to the desired module folder and open the notebook. Alternatively, open any `.ipynb` file directly in **VS Code** using the built-in Jupyter extension.

**Per-module dependency tip:** If a notebook fails with an `ImportError`, install the missing package inside the notebook itself:

```python
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "missing-package"])
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError` for a LangChain package | Run `pip install <package>` inside your active virtual environment. |
| API key errors / `AuthenticationError` | Ensure your `.env` is loaded with `load_dotenv()` and the key name matches what the provider expects. |
| Jupyter kernel not found | Run `python -m ipykernel install --user --name myvenv` to register your venv as a Jupyter kernel. |
| Submodule directory is empty | Run `git submodule update --init` from the repository root. |
| `chromadb` / vector store errors | Make sure `chromadb` is installed and you are using a compatible Python version (3.10+). |
| Streamlit app won't open | Confirm `streamlit` is installed and no other process is using port 8501. Try `streamlit run app.py --server.port 8502`. |
| Rate limit errors from LLM provider | Add retry logic or reduce request frequency. Consider using a different provider or model. |

---

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository on GitHub.
2. Create a feature branch: `git checkout -b feature/my-improvement`
3. Make your changes and commit with a clear message.
4. Push the branch and open a Pull Request against `main`.

Please ensure your code follows these conventions:
- Keep each module self-contained in its numbered directory.
- Do not commit API keys, `.env` files, or large binary files.
- Add or update module-level `README.md` files when introducing a new concept.
- For new modules, follow the existing numbering scheme.

---

## License

This project does not currently specify a license. Please contact the repository owner [@Sikandarh11](https://github.com/Sikandarh11) before using this code in your own projects.

---

*Happy learning! If you find this repository useful, consider giving it a ⭐ on [GitHub](https://github.com/Sikandarh11/Generative-AI-using-LangChain).*
