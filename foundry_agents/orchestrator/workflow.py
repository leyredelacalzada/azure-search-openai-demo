"""
Orchestrator workflow for routing queries to specialized agents.

This module implements a workflow that:
1. Analyzes incoming user queries
2. Routes them to the appropriate specialist agent (Benefits, HR Policy, or Perks)
3. Returns the specialist's response

All agents are grounded to the same Foundry IQ Knowledge Base (gptkbindex-agent-upgrade)
which contains the Contoso HR documents. Each agent specializes in a different domain.

The orchestration uses Microsoft Agent Framework's WorkflowBuilder for
defining the agent graph and managing message flow.
"""

import asyncio
import os
from typing import Literal

from agent_framework import (
    ChatAgent,
    ChatMessage,
    Executor,
    Role,
    WorkflowBuilder,
    WorkflowContext,
    WorkflowOutputEvent,
    WorkflowStatusEvent,
    WorkflowRunState,
    handler,
)
from agent_framework.azure import AzureAIClient, AzureAISearchContextProvider
from azure.identity.aio import DefaultAzureCredential
from typing_extensions import Never

from agents.benefits_agent import get_benefits_agent_instructions
from agents.hr_policy_agent import get_hr_policy_agent_instructions
from agents.perks_agent import get_perks_agent_instructions
from config import load_config

ORCHESTRATOR_INSTRUCTIONS = """You are an HR Assistant Orchestrator. Your role is to analyze employee questions and route them to the appropriate specialist agent.

You have access to three specialist agents:
1. **benefits-agent**: Handles health insurance, medical plans, coverage, deductibles, copays
2. **hr-policy-agent**: Handles workplace policies, employee handbook, job roles, procedures
3. **perks-agent**: Handles wellness programs, gym benefits, vacation, employee recognition, company culture

IMPORTANT: You must respond with ONLY the agent name (one of: benefits-agent, hr-policy-agent, perks-agent) followed by a brief reason.

Format your response as:
AGENT: <agent-name>
REASON: <brief explanation>

Examples:
- Question: "What's my deductible for the health plan?"
  AGENT: benefits-agent
  REASON: Health insurance deductible is a benefits question

- Question: "What's the policy on remote work?"
  AGENT: hr-policy-agent
  REASON: Remote work policy is in the employee handbook

- Question: "How do I use my gym membership benefit?"
  AGENT: perks-agent
  REASON: Gym membership is a wellness perk

If a question spans multiple domains, route to the most relevant agent based on the primary focus.
"""


class OrchestratorAgent(Executor):
    """
    Orchestrator that routes queries to the appropriate specialist agent.
    
    This executor analyzes incoming queries and determines which specialist
    agent is best suited to answer the question.
    """

    agent: ChatAgent

    def __init__(
        self,
        client: AzureAIClient,
        id: str = "orchestrator",
    ):
        """
        Initialize the Orchestrator.
        
        Args:
            client: Azure AI client for model inference
            id: Executor identifier
        """
        self.agent = client.create_agent(
            name="HROrchestrator",
            instructions=ORCHESTRATOR_INSTRUCTIONS,
        )
        super().__init__(id=id)

    def parse_routing_decision(self, response_text: str) -> str:
        """
        Parse the orchestrator's routing decision from its response.
        
        Args:
            response_text: The orchestrator's response text
            
        Returns:
            The agent ID to route to (defaults to benefits-agent if parsing fails)
        """
        response_lower = response_text.lower()
        
        # Look for AGENT: pattern
        if "agent:" in response_lower:
            for line in response_text.split("\n"):
                if "agent:" in line.lower():
                    agent_part = line.split(":", 1)[1].strip().lower()
                    if "benefits" in agent_part:
                        return "benefits-agent"
                    elif "hr-policy" in agent_part or "policy" in agent_part:
                        return "hr-policy-agent"
                    elif "perks" in agent_part:
                        return "perks-agent"
        
        # Fallback: look for agent names anywhere in the response
        if "benefits" in response_lower:
            return "benefits-agent"
        elif "hr-policy" in response_lower or "policy" in response_lower:
            return "hr-policy-agent"
        elif "perks" in response_lower:
            return "perks-agent"
        
        # Default to benefits agent
        return "benefits-agent"

    @handler
    async def route_query(
        self, message: ChatMessage, ctx: WorkflowContext[tuple[str, ChatMessage]]
    ) -> None:
        """
        Analyze a query and route it to the appropriate specialist.
        
        Args:
            message: The user's query
            ctx: Workflow context for sending to the next executor
        """
        # Ask the orchestrator which agent should handle this query
        routing_response = await self.agent.run([message])
        target_agent = self.parse_routing_decision(routing_response.text)
        
        print(f"\n🔀 Routing to: {target_agent}")
        print(f"   Reason: {routing_response.text.split('REASON:')[-1].strip() if 'REASON:' in routing_response.text else 'N/A'}")
        
        # Send the original message to the selected specialist
        await ctx.send_message((target_agent, message))


class SpecialistRouter(Executor):
    """
    Router that receives routing decisions and forwards to the correct specialist.
    """

    specialists: dict[str, ChatAgent]

    def __init__(
        self,
        specialists: dict[str, ChatAgent],
        id: str = "router",
    ):
        """
        Initialize the router.
        
        Args:
            specialists: Dictionary mapping agent IDs to ChatAgent instances
            id: Executor identifier
        """
        self.specialists = specialists
        super().__init__(id=id)

    @handler
    async def handle_routed_message(
        self, routing_data: tuple[str, ChatMessage], ctx: WorkflowContext[Never, str]
    ) -> None:
        """
        Forward the message to the appropriate specialist and yield the response.
        
        Args:
            routing_data: Tuple of (agent_id, message)
            ctx: Workflow context for yielding output
        """
        agent_id, message = routing_data
        
        if agent_id not in self.specialists:
            await ctx.yield_output(f"Error: Unknown agent '{agent_id}'")
            return
        
        specialist = self.specialists[agent_id]
        response = await specialist.run([message])
        await ctx.yield_output(response.text)


def get_kb_variant() -> str:
    """Get the Knowledge Base variant to use from environment."""
    variant = os.environ.get("KNOWLEDGE_BASE_VARIANT", "base")
    valid_variants = ["base", "with-sharepoint", "with-web", "with-web-and-sharepoint"]
    if variant not in valid_variants:
        variant = "base"
    return variant


async def create_hr_assistant_workflow(
    config=None,
    kb_variant: str | None = None,
) -> tuple:
    """
    Create the HR Assistant workflow with all agents and context providers.
    
    This sets up:
    - Azure AI Search context provider for the existing Foundry IQ Knowledge Base
    - Specialist agents all grounded to the same KB (with different instructions)
    - Orchestrator agent for routing
    - Workflow connecting all components
    
    Args:
        config: Optional configuration
        kb_variant: Which KB variant to use (base, with-sharepoint, with-web, with-web-and-sharepoint)
    
    Returns:
        Tuple of (workflow, credential, client, context_provider)
    """
    if config is None:
        config = load_config()
    
    if kb_variant is None:
        kb_variant = get_kb_variant()
    
    # Get the appropriate Knowledge Base configuration
    kb_config = config.knowledge_bases.get(kb_variant, config.knowledge_bases["base"])
    
    print(f"\n📚 Using Knowledge Base: {kb_config.name}")
    print(f"   Data sources: {', '.join(kb_config.data_sources)}")
    
    credential = DefaultAzureCredential()
    
    # Create the Azure AI client
    client = AzureAIClient(
        project_endpoint=config.foundry_project_endpoint,
        model_deployment_name=config.openai_deployment,
        credential=credential,
    )
    
    # Create a single context provider for the existing Knowledge Base
    # All agents share this KB but have different instructions
    context_provider = AzureAISearchContextProvider(
        endpoint=config.search_endpoint,
        knowledge_base_name=kb_config.name,
        credential=credential,
        mode=config.retrieval_mode,
        azure_openai_resource_url=config.openai_endpoint if config.retrieval_mode == "agentic" else None,
        model_deployment_name=config.openai_deployment if config.retrieval_mode == "agentic" else None,
        retrieval_reasoning_effort=config.retrieval_reasoning_effort if config.retrieval_mode == "agentic" else None,
    )
    
    # Create specialist agents - all grounded to the same KB but with different instructions
    benefits_agent = client.create_agent(
        name="BenefitsAgent",
        instructions=get_benefits_agent_instructions(),
        context_providers=[context_provider],
    )
    
    hr_policy_agent = client.create_agent(
        name="HRPolicyAgent",
        instructions=get_hr_policy_agent_instructions(),
        context_providers=[context_provider],
    )
    
    perks_agent = client.create_agent(
        name="PerksAgent",
        instructions=get_perks_agent_instructions(),
        context_providers=[context_provider],
    )
    
    specialists = {
        "benefits-agent": benefits_agent,
        "hr-policy-agent": hr_policy_agent,
        "perks-agent": perks_agent,
    }
    
    # Create executors
    orchestrator = OrchestratorAgent(client, id="orchestrator")
    router = SpecialistRouter(specialists, id="router")
    
    # Build the workflow
    # User -> Orchestrator -> Router -> (yields output)
    workflow = (
        WorkflowBuilder()
        .set_start_executor(orchestrator)
        .add_edge(orchestrator, router)
        .build()
    )
    
    return workflow, credential, client, context_provider


async def run_hr_assistant(query: str, config=None, kb_variant: str | None = None) -> str:
    """
    Run a single query through the HR Assistant workflow.
    
    Args:
        query: The user's question
        config: Optional configuration (uses default if not provided)
        kb_variant: Which KB variant to use
        
    Returns:
        The specialist agent's response
    """
    workflow, credential, client, context_provider = await create_hr_assistant_workflow(config, kb_variant)
    
    try:
        message = ChatMessage(role=Role.USER, text=query)
        result = None
        
        async for event in workflow.run_stream(message):
            if isinstance(event, WorkflowStatusEvent):
                if event.state == WorkflowRunState.IDLE:
                    break
            elif isinstance(event, WorkflowOutputEvent):
                result = event.data
        
        return result or "No response received from the workflow."
    
    finally:
        # Cleanup
        await credential.close()


async def interactive_session(config=None, kb_variant: str | None = None):
    """
    Run an interactive session with the HR Assistant.
    
    This allows users to ask multiple questions in a loop.
    """
    if config is None:
        config = load_config()
    
    if kb_variant is None:
        kb_variant = get_kb_variant()
    
    print("\n" + "=" * 60)
    print("🤖 HR Assistant - Powered by Foundry IQ + Agent Framework")
    print("=" * 60)
    print("\nI can help you with:")
    print("  • Health benefits and insurance plans")
    print("  • Workplace policies and procedures")
    print("  • Employee perks and wellness programs")
    print("\nType 'quit' or 'exit' to end the session.\n")
    
    workflow, credential, client, context_provider = await create_hr_assistant_workflow(config, kb_variant)
    
    try:
        while True:
            try:
                query = input("\n❓ Your question: ").strip()
            except EOFError:
                break
            
            if not query:
                continue
            
            if query.lower() in ("quit", "exit", "q"):
                print("\n👋 Goodbye!")
                break
            
            print("\n⏳ Processing...")
            
            message = ChatMessage(role=Role.USER, text=query)
            
            async for event in workflow.run_stream(message):
                if isinstance(event, WorkflowStatusEvent):
                    if event.state == WorkflowRunState.IDLE:
                        break
                elif isinstance(event, WorkflowOutputEvent):
                    print(f"\n💬 Response:\n{event.data}")
    
    finally:
        await credential.close()
        # Give time for async cleanup
        await asyncio.sleep(0.5)
