"""
Configuration management for the Foundry IQ + Agent Framework demo.
"""

import os
from dataclasses import dataclass
from typing import Literal

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class KnowledgeBaseConfig:
    """Configuration for a Knowledge Base."""

    name: str
    description: str
    index_name: str
    data_sources: list[str]


@dataclass
class Config:
    """Application configuration."""

    # Azure AI Search
    search_endpoint: str
    
    # Azure OpenAI
    openai_endpoint: str
    openai_deployment: str
    
    # Microsoft Foundry
    foundry_project_endpoint: str
    
    # Retrieval settings
    retrieval_mode: Literal["semantic", "agentic"]
    retrieval_reasoning_effort: Literal["low", "medium", "high"]
    
    # Knowledge Base configurations
    knowledge_bases: dict[str, KnowledgeBaseConfig]


def get_config() -> Config:
    """Load configuration from environment variables."""
    
    # Define Knowledge Base configurations using existing KBs from Azure AI Search
    # These KBs are already created in the gptkb-j2f5gccswaftm search resource
    knowledge_bases = {
        # Base KB with just the search index (Contoso HR documents)
        "base": KnowledgeBaseConfig(
            name="gptkbindex-agent-upgrade",
            description="Contoso HR knowledge base with employee handbook, benefits, and policies",
            index_name="gptkbindex",
            data_sources=["gptkbindex"],
        ),
        # KB with SharePoint integration
        "with-sharepoint": KnowledgeBaseConfig(
            name="gptkbindex-agent-upgrade-with-sp",
            description="Contoso HR knowledge plus SharePoint documents",
            index_name="gptkbindex",
            data_sources=["gptkbindex", "sharepoint"],
        ),
        # KB with web search integration
        "with-web": KnowledgeBaseConfig(
            name="gptkbindex-agent-upgrade-with-web",
            description="Contoso HR knowledge plus web search capabilities",
            index_name="gptkbindex",
            data_sources=["gptkbindex", "web"],
        ),
        # KB with both web and SharePoint
        "with-web-and-sharepoint": KnowledgeBaseConfig(
            name="gptkbindex-agent-upgrade-with-web-and-sp",
            description="Contoso HR knowledge plus web search and SharePoint",
            index_name="gptkbindex",
            data_sources=["gptkbindex", "web", "sharepoint"],
        ),
    }
    
    retrieval_mode = os.getenv("RETRIEVAL_MODE", "agentic")
    if retrieval_mode not in ("semantic", "agentic"):
        retrieval_mode = "agentic"
    
    reasoning_effort = os.getenv("RETRIEVAL_REASONING_EFFORT", "medium")
    if reasoning_effort not in ("low", "medium", "high"):
        reasoning_effort = "medium"
    
    return Config(
        search_endpoint=os.environ.get("AZURE_SEARCH_ENDPOINT", "https://gptkb-j2f5gccswaftm.search.windows.net"),
        openai_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT", ""),
        openai_deployment=os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
        foundry_project_endpoint=os.environ.get("FOUNDRY_PROJECT_ENDPOINT", ""),
        retrieval_mode=retrieval_mode,  # type: ignore
        retrieval_reasoning_effort=reasoning_effort,  # type: ignore
        knowledge_bases=knowledge_bases,
    )


# Singleton config instance
_config: Config | None = None


def load_config() -> Config:
    """Load and cache configuration."""
    global _config
    if _config is None:
        _config = get_config()
    return _config
