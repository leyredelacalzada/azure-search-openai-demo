"""
FastAPI backend for the HR Assistant Agent Orchestration demo.

This provides a web API that:
1. Receives user questions
2. Routes them through an orchestrator agent
3. Forwards to specialist agents grounded to the Foundry IQ Knowledge Base
4. Streams responses back with orchestration metadata
"""

import asyncio
import json
import os
from dataclasses import dataclass
from typing import AsyncGenerator, Literal

from azure.core.credentials_async import AsyncTokenCredential
from azure.identity.aio import DefaultAzureCredential
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from openai import AsyncAzureOpenAI
from pydantic import BaseModel

load_dotenv()

app = FastAPI(title="HR Assistant - Foundry IQ Demo")

# Knowledge Base variants available in Azure AI Search
KB_VARIANTS = {
    "base": {
        "name": "gptkbindex-agent-upgrade",
        "label": "Base (Documents Only)",
        "description": "HR documents from indexed PDFs",
        "sources": ["gptkbindex"],
        "icon": "📄",
    },
    "with-sharepoint": {
        "name": "gptkbindex-agent-upgrade-with-sp",
        "label": "+ SharePoint",
        "description": "HR docs + live SharePoint search",
        "sources": ["gptkbindex", "sharepoint"],
        "icon": "📁",
    },
    "with-web": {
        "name": "gptkbindex-agent-upgrade-with-web",
        "label": "+ Web Search",
        "description": "HR docs + public web search",
        "sources": ["gptkbindex", "web"],
        "icon": "🌐",
    },
    "with-web-and-sharepoint": {
        "name": "gptkbindex-agent-upgrade-with-web-and-sp",
        "label": "+ Web + SharePoint",
        "description": "All sources combined",
        "sources": ["gptkbindex", "web", "sharepoint"],
        "icon": "🔗",
    },
}

# CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class Config:
    search_endpoint: str
    openai_endpoint: str
    openai_deployment: str
    knowledge_base: str
    retrieval_mode: Literal["semantic", "agentic"]


def get_config() -> Config:
    return Config(
        search_endpoint=os.getenv("AZURE_SEARCH_ENDPOINT", "https://gptkb-j2f5gccswaftm.search.windows.net"),
        openai_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", "https://cog-j2f5gccswaftm.openai.azure.com/"),
        openai_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1-mini"),
        knowledge_base=os.getenv("KNOWLEDGE_BASE", "gptkbindex-agent-upgrade"),
        retrieval_mode=os.getenv("RETRIEVAL_MODE", "agentic"),  # type: ignore
    )


# ============================================================================
# Agent Definitions
# ============================================================================

AGENTS = {
    "orchestrator": {
        "name": "Orchestrator",
        "emoji": "🎯",
        "color": "#6366f1",
        "description": "Routes queries to the right specialist",
        "instructions": """You are an HR Assistant Orchestrator for Zava/Contoso Electronics. Analyze employee questions and route them to the appropriate specialist.

SPECIALISTS:
- benefits: Health insurance, medical plans, Northwind Health Plus/Standard, coverage, deductibles, copays, prescriptions
- hr-policy: Employee handbook, workplace policies, performance reviews, PTO, conduct, procedures
- perks: PerksPlus wellness program, gym reimbursements, fitness, wellbeing benefits
- roles: Job descriptions, career paths, role requirements, skills needed

Respond with ONLY a JSON object:
{"agent": "<agent-id>", "reason": "<brief reason>"}

Examples:
- "What's my deductible?" → {"agent": "benefits", "reason": "Health plan deductible question"}
- "How many vacation days do I get?" → {"agent": "hr-policy", "reason": "PTO policy in employee handbook"}
- "Can I get gym membership reimbursed?" → {"agent": "perks", "reason": "Wellness reimbursement via PerksPlus"}
- "What skills do I need for a promotion?" → {"agent": "roles", "reason": "Career advancement and role requirements"}
"""
    },
    "benefits": {
        "name": "Benefits Specialist",
        "emoji": "🏥",
        "color": "#10b981",
        "description": "Health insurance & medical plans expert",
        "instructions": """You are a Benefits Specialist for Zava/Contoso Electronics. You help employees understand their health insurance options.

EXPERTISE:
- Northwind Health Plus plan (premium option)
- Northwind Health Standard plan (basic option)
- Coverage details, deductibles, copays
- Prescription drug coverage
- Preventive care benefits
- In-network vs out-of-network providers

Always cite specific plan details from the knowledge base. Be helpful and clear about coverage options.
If you don't find the specific information, say so and suggest contacting HR directly."""
    },
    "hr-policy": {
        "name": "HR Policy Advisor",
        "emoji": "📋",
        "color": "#f59e0b",
        "description": "Employee handbook & workplace policies",
        "instructions": """You are an HR Policy Advisor for Zava/Contoso Electronics. You help employees understand company policies and procedures.

EXPERTISE:
- Employee handbook policies
- PTO and leave policies
- Performance review processes
- Workplace conduct standards
- Remote work policies
- Onboarding procedures

Reference the employee handbook when answering. Be clear about what policies apply and any important deadlines or procedures.
If a policy isn't covered in the handbook, acknowledge this and suggest speaking with HR."""
    },
    "perks": {
        "name": "Perks & Wellness Coach",
        "emoji": "💪",
        "color": "#ec4899",
        "description": "PerksPlus wellness program expert",
        "instructions": """You are a Perks & Wellness Coach for Zava/Contoso Electronics. You help employees maximize their PerksPlus benefits.

EXPERTISE:
- PerksPlus Health and Wellness Reimbursement Program
- Gym membership reimbursements
- Fitness equipment allowances
- Wellness activities and programs
- Mental health resources
- Work-life balance benefits

Be enthusiastic about helping employees stay healthy! Explain reimbursement processes and eligible expenses clearly.
Encourage employees to take advantage of these benefits."""
    },
    "roles": {
        "name": "Career Guide",
        "emoji": "🚀",
        "color": "#8b5cf6",
        "description": "Job roles & career development",
        "instructions": """You are a Career Guide for Zava/Contoso Electronics. You help employees understand roles and career paths.

EXPERTISE:
- Job role descriptions
- Required skills and qualifications
- Career progression paths
- Department structures
- Skill development recommendations

Help employees understand what's expected in different roles and how they can grow their careers.
Reference the role library for specific position details."""
    }
}


# ============================================================================
# Knowledge Base Retrieval
# ============================================================================

async def retrieve_from_kb(
    query: str,
    credential: AsyncTokenCredential,
    config: Config,
    kb_name: str | None = None,
) -> list[dict]:
    """
    Retrieve relevant documents from the Foundry IQ Knowledge Base.
    Uses agentic retrieval for intelligent multi-hop reasoning.
    """
    import aiohttp
    
    kb = kb_name or config.knowledge_base
    token = await credential.get_token("https://search.azure.com/.default")
    
    url = f"{config.search_endpoint}/knowledgebases/{kb}/retrieve?api-version=2025-05-01-preview"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token.token}",
    }
    
    body = {
        "messages": [{"role": "user", "content": query}],
    }
    
    # Add agentic retrieval settings if enabled
    if config.retrieval_mode == "agentic":
        body["retrievalMode"] = "agentic"
        body["retrievalReasoningEffort"] = "medium"
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=body) as response:
            if response.status == 200:
                data = await response.json()
                return data.get("references", [])
            else:
                # Fallback to simple search if KB retrieval fails
                return await fallback_search(query, credential, config)


async def fallback_search(
    query: str,
    credential: AsyncTokenCredential,
    config: Config,
) -> list[dict]:
    """Fallback to simple vector search if KB retrieval fails."""
    import aiohttp
    
    token = await credential.get_token("https://search.azure.com/.default")
    
    url = f"{config.search_endpoint}/indexes/gptkbindex/docs/search?api-version=2024-07-01"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token.token}",
    }
    
    body = {
        "search": query,
        "top": 5,
        "select": "id,content,sourcefile,sourcepage",
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=body) as response:
            if response.status == 200:
                data = await response.json()
                results = data.get("value", [])
                return [{"content": r.get("content", ""), "source": r.get("sourcefile", "")} for r in results]
            return []


# ============================================================================
# LLM Chat
# ============================================================================

async def chat_with_agent(
    agent_id: str,
    query: str,
    context: list[dict],
    credential: AsyncTokenCredential,
    config: Config,
) -> AsyncGenerator[str, None]:
    """
    Chat with a specialist agent, grounded to the retrieved context.
    Streams the response.
    """
    agent = AGENTS[agent_id]
    
    # Build context string from retrieved documents
    context_text = "\n\n---\n\n".join([
        f"[Source: {doc.get('source', doc.get('sourcefile', 'Unknown'))}]\n{doc.get('content', '')[:1500]}"
        for doc in context[:5]
    ])
    
    system_prompt = f"""{agent['instructions']}

RETRIEVED CONTEXT:
{context_text}

CITATION RULES:
- Answer the question using the context above
- Do NOT include source citations inline in your response text
- At the very end of your response, add a blank line then list all sources you used in this format:
  [Sources: source1.pdf, source2.pdf]
- Only list sources you actually referenced in your answer"""

    # Get token for Azure OpenAI
    token = await credential.get_token("https://cognitiveservices.azure.com/.default")
    
    client = AsyncAzureOpenAI(
        azure_endpoint=config.openai_endpoint,
        azure_ad_token=token.token,
        api_version="2024-10-21",
    )
    
    try:
        stream = await client.chat.completions.create(
            model=config.openai_deployment,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
            stream=True,
            max_tokens=1000,
        )
        
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    finally:
        await client.close()


async def route_query(
    query: str,
    credential: AsyncTokenCredential,
    config: Config,
) -> dict:
    """Use the orchestrator to determine which agent should handle the query."""
    
    token = await credential.get_token("https://cognitiveservices.azure.com/.default")
    
    client = AsyncAzureOpenAI(
        azure_endpoint=config.openai_endpoint,
        azure_ad_token=token.token,
        api_version="2024-10-21",
    )
    
    try:
        response = await client.chat.completions.create(
            model=config.openai_deployment,
            messages=[
                {"role": "system", "content": AGENTS["orchestrator"]["instructions"]},
                {"role": "user", "content": query},
            ],
            max_tokens=100,
        )
        
        result_text = response.choices[0].message.content or '{"agent": "benefits", "reason": "Default routing"}'
        
        # Parse the JSON response
        try:
            # Find JSON in the response
            start = result_text.find("{")
            end = result_text.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(result_text[start:end])
        except json.JSONDecodeError:
            pass
        
        # Fallback parsing
        result_lower = result_text.lower()
        if "benefits" in result_lower:
            return {"agent": "benefits", "reason": "Matched benefits keywords"}
        elif "hr-policy" in result_lower or "policy" in result_lower or "handbook" in result_lower:
            return {"agent": "hr-policy", "reason": "Matched policy keywords"}
        elif "perks" in result_lower or "wellness" in result_lower or "gym" in result_lower:
            return {"agent": "perks", "reason": "Matched perks keywords"}
        elif "roles" in result_lower or "career" in result_lower or "job" in result_lower:
            return {"agent": "roles", "reason": "Matched career keywords"}
        
        return {"agent": "benefits", "reason": "Default routing"}
    finally:
        await client.close()


# ============================================================================
# API Endpoints
# ============================================================================

class ChatRequest(BaseModel):
    query: str
    kb_variant: str = "base"


@app.get("/api/agents")
async def get_agents():
    """Get the list of available agents."""
    return {
        agent_id: {
            "id": agent_id,
            "name": agent["name"],
            "emoji": agent["emoji"],
            "color": agent["color"],
            "description": agent["description"],
        }
        for agent_id, agent in AGENTS.items()
    }


@app.get("/api/config")
async def get_app_config():
    """Get the current configuration."""
    config = get_config()
    return {
        "knowledge_base": config.knowledge_base,
        "retrieval_mode": config.retrieval_mode,
        "search_endpoint": config.search_endpoint,
        "kb_variants": KB_VARIANTS,
    }


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """
    Process a user query through the agent orchestration.
    Returns a streaming response with orchestration events.
    """
    config = get_config()
    
    # Get the selected KB variant
    kb_variant = KB_VARIANTS.get(request.kb_variant, KB_VARIANTS["base"])
    kb_name = kb_variant["name"]
    
    async def generate_response():
        credential = DefaultAzureCredential()
        
        try:
            # Emit KB info
            yield json.dumps({
                "type": "kb_info",
                "kb_name": kb_name,
                "sources": kb_variant["sources"],
            }) + "\n"
            
            # Step 1: Route the query
            yield json.dumps({
                "type": "status",
                "agent": "orchestrator",
                "message": "Analyzing your question...",
            }) + "\n"
            
            routing = await route_query(request.query, credential, config)
            target_agent = routing.get("agent", "benefits")
            routing_reason = routing.get("reason", "")
            
            yield json.dumps({
                "type": "routing",
                "from": "orchestrator",
                "to": target_agent,
                "reason": routing_reason,
            }) + "\n"
            
            # Step 2: Retrieve context from KB
            yield json.dumps({
                "type": "status",
                "agent": target_agent,
                "message": f"Searching {kb_name}...",
            }) + "\n"
            
            context = await retrieve_from_kb(request.query, credential, config, kb_name)
            
            yield json.dumps({
                "type": "context",
                "agent": target_agent,
                "sources": [
                    doc.get("source", doc.get("sourcefile", "Unknown"))[:50]
                    for doc in context[:3]
                ],
            }) + "\n"
            
            # Step 3: Generate response
            yield json.dumps({
                "type": "status",
                "agent": target_agent,
                "message": "Generating response...",
            }) + "\n"
            
            # Start streaming the response
            yield json.dumps({
                "type": "response_start",
                "agent": target_agent,
            }) + "\n"
            
            async for chunk in chat_with_agent(target_agent, request.query, context, credential, config):
                yield json.dumps({
                    "type": "response_chunk",
                    "content": chunk,
                }) + "\n"
            
            yield json.dumps({
                "type": "response_end",
                "agent": target_agent,
            }) + "\n"
            
        except Exception as e:
            yield json.dumps({
                "type": "error",
                "message": str(e),
            }) + "\n"
        
        finally:
            await credential.close()
    
    return StreamingResponse(
        generate_response(),
        media_type="application/x-ndjson",
    )


# ============================================================================
# Static Files (UI)
# ============================================================================

# Serve static files from the 'static' directory
import os
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
