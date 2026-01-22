"""
Benefits Agent - Specialist for health insurance and benefits questions.

This agent is grounded to the benefits Knowledge Base containing:
- Benefit_Options.pdf
- Northwind_Health_Plus_Benefits_Details.pdf
- Northwind_Standard_Benefits_Details.pdf
"""

from agent_framework import (
    ChatAgent,
    ChatMessage,
    Executor,
    WorkflowContext,
    handler,
)
from typing_extensions import Never

BENEFITS_AGENT_INSTRUCTIONS = """You are a Benefits Specialist Agent for the company's HR department.

Your expertise covers:
- Health insurance plans (Northwind Health Plus and Northwind Standard)
- Plan comparisons and coverage details
- Deductibles, copays, and out-of-pocket maximums
- Enrollment processes and eligibility
- Prescription drug coverage
- Vision and dental benefits

IMPORTANT GUIDELINES:
1. Always provide accurate information based on the knowledge base
2. When comparing plans, be clear about the differences
3. If you don't have specific information, say so clearly
4. Cite the source document when referencing specific details
5. Be helpful and empathetic - benefits decisions are important to employees

When answering questions:
- Start with a direct answer to the question
- Provide relevant details and context
- If applicable, suggest related information the employee might find useful
"""


class BenefitsAgentExecutor(Executor):
    """
    Executor that wraps the Benefits Agent for use in workflows.
    
    This executor handles questions about health insurance plans and benefits,
    using Foundry IQ to retrieve relevant information from benefits documents.
    """

    agent: ChatAgent

    def __init__(self, agent: ChatAgent, id: str = "benefits-agent"):
        """
        Initialize the Benefits Agent Executor.
        
        Args:
            agent: A ChatAgent instance grounded to the benefits Knowledge Base
            id: Executor identifier for workflow routing
        """
        self.agent = agent
        super().__init__(id=id)

    @handler
    async def handle_query(
        self, message: ChatMessage, ctx: WorkflowContext[Never, str]
    ) -> None:
        """
        Handle a benefits-related query and yield the response.
        
        Args:
            message: The incoming user message
            ctx: Workflow context for yielding output
        """
        response = await self.agent.run([message])
        await ctx.yield_output(response.text)

    @handler
    async def handle_conversation(
        self, messages: list[ChatMessage], ctx: WorkflowContext[Never, str]
    ) -> None:
        """
        Handle a conversation (multiple messages) and yield the response.
        
        Args:
            messages: The conversation history
            ctx: Workflow context for yielding output
        """
        response = await self.agent.run(messages)
        await ctx.yield_output(response.text)


def get_benefits_agent_instructions() -> str:
    """Return the system instructions for the Benefits Agent."""
    return BENEFITS_AGENT_INSTRUCTIONS
