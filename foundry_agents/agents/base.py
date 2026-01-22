"""
Base agent utilities for creating agents grounded to Foundry IQ Knowledge Bases.
"""

from typing import Literal

from agent_framework import ChatAgent
from agent_framework.azure import AzureAIClient, AzureAISearchContextProvider
from azure.identity.aio import DefaultAzureCredential


async def create_search_context_provider(
    search_endpoint: str,
    knowledge_base_name: str,
    credential: DefaultAzureCredential,
    mode: Literal["semantic", "agentic"] = "agentic",
    openai_endpoint: str | None = None,
    openai_deployment: str | None = None,
    reasoning_effort: Literal["low", "medium", "high"] = "medium",
) -> AzureAISearchContextProvider:
    """
    Create an Azure AI Search context provider for Foundry IQ.
    
    Args:
        search_endpoint: Azure AI Search endpoint URL
        knowledge_base_name: Name of the Foundry IQ Knowledge Base
        credential: Azure credential for authentication
        mode: Retrieval mode - "semantic" for fast hybrid search, "agentic" for intelligent retrieval
        openai_endpoint: Azure OpenAI endpoint (required for agentic mode)
        openai_deployment: Azure OpenAI deployment name (required for agentic mode)
        reasoning_effort: Query planning effort for agentic mode (low, medium, high)
    
    Returns:
        Configured AzureAISearchContextProvider
    """
    if mode == "agentic":
        return AzureAISearchContextProvider(
            endpoint=search_endpoint,
            knowledge_base_name=knowledge_base_name,
            credential=credential,
            mode="agentic",
            azure_openai_resource_url=openai_endpoint,
            model_deployment_name=openai_deployment,
            retrieval_reasoning_effort=reasoning_effort,
        )
    else:
        return AzureAISearchContextProvider(
            endpoint=search_endpoint,
            knowledge_base_name=knowledge_base_name,
            credential=credential,
            mode="semantic",
            top_k=5,
        )


def create_grounded_agent(
    client: AzureAIClient,
    name: str,
    instructions: str,
    context_providers: list,
) -> ChatAgent:
    """
    Create a ChatAgent grounded to Foundry IQ Knowledge Bases via context providers.
    
    Args:
        client: Azure AI client for model inference
        name: Agent name (must be alphanumeric with optional hyphens, max 63 chars)
        instructions: System instructions for the agent
        context_providers: List of context providers (e.g., AzureAISearchContextProvider)
    
    Returns:
        Configured ChatAgent
    """
    return client.create_agent(
        name=name,
        instructions=instructions,
        context_providers=context_providers,
    )
