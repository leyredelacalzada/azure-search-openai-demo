# Foundry IQ + Agent Framework Demo

This demo showcases how to integrate Microsoft Agent Framework with **existing** Foundry IQ Knowledge Bases for intelligent, multi-agent orchestration.

## Overview

This project extends the Azure Search OpenAI demo by adding:

1. **Existing Knowledge Bases** - Uses pre-configured Foundry IQ Knowledge Bases from Azure AI Search
2. **Specialized Agents** - 3 agents (Benefits, HR Policy, Perks), each with domain-specific instructions
3. **Agent Orchestration** - A workflow that routes queries to the appropriate agent based on intent

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    User Query                                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Orchestrator Agent                            │
│  (Analyzes query intent and routes to specialist)               │
└─────────────────────────────────────────────────────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         ▼                    ▼                    ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│  Benefits Agent │ │ HR Policy Agent │ │   Perks Agent   │
│  (health plans) │ │   (handbook)    │ │   (wellness)    │
└─────────────────┘ └─────────────────┘ └─────────────────┘
         │                    │                    │
         └────────────────────┼────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              Foundry IQ Knowledge Base                           │
│              (gptkbindex-agent-upgrade)                          │
│                                                                  │
│  Data Sources: gptkbindex, web (optional), sharepoint (optional) │
└─────────────────────────────────────────────────────────────────┘
```

## Available Knowledge Bases

The demo uses existing Knowledge Bases from the `gptkb-j2f5gccswaftm` Azure AI Search resource:

| Variant | Knowledge Base Name | Data Sources |
|---------|---------------------|--------------|
| `base` | gptkbindex-agent-upgrade | gptkbindex (HR documents) |
| `with-sharepoint` | gptkbindex-agent-upgrade-with-sp | gptkbindex + SharePoint |
| `with-web` | gptkbindex-agent-upgrade-with-web | gptkbindex + Web search |
| `with-web-and-sharepoint` | gptkbindex-agent-upgrade-with-web-and-sp | gptkbindex + Web + SharePoint |

## Agents

All agents share the same Knowledge Base but have specialized instructions:

| Agent | Specialization | Example Questions |
|-------|----------------|-------------------|
| **Benefits Agent** | Health insurance, coverage, costs | "What's the deductible for the health plan?" |
| **HR Policy Agent** | Workplace policies, procedures | "What's the policy on remote work?" |
| **Perks Agent** | Wellness programs, vacation | "What gym benefits are available?" |

## Prerequisites

1. **Azure Subscription** with:
   - Azure AI Search service (gptkb-j2f5gccswaftm)
   - Azure OpenAI service (with gpt-4.1-mini deployed)
   - Azure AI Foundry project

2. **Python 3.10+**

3. **Azure CLI** logged in (`az login`)

## Setup

### 1. Install Dependencies

```bash
cd foundry_agents
pip install -r requirements.txt
```

### 2. Configure Environment

Copy the sample environment file and update with your values:

```bash
cp .env.sample .env
```

Edit `.env`:

```env
AZURE_SEARCH_ENDPOINT=https://gptkb-j2f5gccswaftm.search.windows.net
AZURE_OPENAI_ENDPOINT=https://your-openai.openai.azure.com
AZURE_OPENAI_DEPLOYMENT=gpt-4.1-mini
FOUNDRY_PROJECT_ENDPOINT=https://your-project.api.azureml.ms
KNOWLEDGE_BASE_VARIANT=base
```

### 3. Run the Demo

```bash
# Interactive mode (default)
python main.py

# With specific KB variant
python main.py --kb with-web

# Run demo queries
python main.py --demo --kb base

# Single query
python main.py -q "What health benefits are available?" --kb with-sharepoint
```

## Usage Examples

```bash
# Basic usage with base KB
python main.py --kb base

# Use KB with web search capabilities
python main.py --kb with-web

# Use KB with SharePoint integration
python main.py --kb with-sharepoint

# Use KB with all data sources
python main.py --kb with-web-and-sharepoint

# Run a single query
python main.py -q "What's the difference between health plan options?" --kb base

# Validate configuration
python main.py --validate
```

## Retrieval Modes

The demo supports two Foundry IQ retrieval modes:

- **`agentic`** (default) - Intelligent multi-hop retrieval with query planning
- **`semantic`** - Fast hybrid search (keyword + vector + semantic reranking)

Set via environment variable:
```env
RETRIEVAL_MODE=agentic
RETRIEVAL_REASONING_EFFORT=medium  # low, medium, high
```

## Project Structure

```
foundry_agents/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── .env.sample                  # Environment template
├── config.py                    # Configuration with existing KB definitions
├── main.py                      # Demo entry point
├── agents/
│   ├── __init__.py
│   ├── base.py                  # Base utilities for context providers
│   ├── benefits_agent.py        # Health benefits specialist
│   ├── hr_policy_agent.py       # HR policies specialist
│   └── perks_agent.py           # Perks and wellness specialist
└── orchestrator/
    ├── __init__.py
    └── workflow.py              # Agent orchestration workflow
```

## Workshop Demo Flow

1. **Show the existing Knowledge Bases** in Azure Portal (AI Search → Knowledge bases)

2. **Explain the architecture**:
   - Single KB shared by multiple specialist agents
   - Orchestrator routes based on query intent
   - Each agent has domain-specific instructions

3. **Run with different KB variants** to show data source flexibility:
   ```bash
   # Base: just HR documents
   python main.py --kb base -q "What health plans are available?"
   
   # With web: can augment with web search
   python main.py --kb with-web -q "What health plans are available?"
   ```

4. **Show agent routing** in action:
   ```bash
   python main.py --demo
   ```

## Learn More

- [Foundry IQ in Microsoft Agent Framework](https://devblogs.microsoft.com/foundry/foundry-iq-agent-framework-integration/)
- [Microsoft Agent Framework](https://aka.ms/agent-framework)
- [Azure AI Search Knowledge Bases](https://learn.microsoft.com/azure/search/agentic-retrieval-how-to-create-knowledge-base)
- [Agentic Retrieval Demo](https://aka.ms/foundry-iq-demo)
