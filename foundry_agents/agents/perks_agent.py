"""
Perks Agent - Specialist for employee perks and wellness benefits.

This agent is grounded to the perks Knowledge Base containing:
- PerksPlus.pdf
- Zava_Company_Overview.md (vacation and recognition sections)
"""

from agent_framework import (
    ChatAgent,
    ChatMessage,
    Executor,
    WorkflowContext,
    handler,
)
from typing_extensions import Never

PERKS_AGENT_INSTRUCTIONS = """You are a Perks and Wellness Specialist Agent for the company.

Your expertise covers:
- Employee perks and benefits beyond health insurance
- Wellness programs and initiatives
- Gym memberships and fitness benefits
- Vacation policies and time-off benefits
- Employee recognition programs
- Work-life balance initiatives
- Professional development allowances
- Company culture and values

IMPORTANT GUIDELINES:
1. Be enthusiastic about the company's perks - they're designed to make employees happy!
2. Provide specific details about how to access or use perks
3. Mention any eligibility requirements or limitations
4. Encourage employees to take advantage of available benefits
5. Be helpful in navigating the various programs

When answering questions:
- Highlight the value of each perk
- Explain enrollment or activation processes
- Mention any time limits or deadlines
- Suggest related perks the employee might not know about
"""


class PerksAgentExecutor(Executor):
    """
    Executor that wraps the Perks Agent for use in workflows.
    
    This executor handles questions about employee perks, wellness programs,
    and work-life balance benefits using Foundry IQ.
    """

    agent: ChatAgent

    def __init__(self, agent: ChatAgent, id: str = "perks-agent"):
        """
        Initialize the Perks Agent Executor.
        
        Args:
            agent: A ChatAgent instance grounded to the perks Knowledge Base
            id: Executor identifier for workflow routing
        """
        self.agent = agent
        super().__init__(id=id)

    @handler
    async def handle_query(
        self, message: ChatMessage, ctx: WorkflowContext[Never, str]
    ) -> None:
        """
        Handle a perks-related query and yield the response.
        
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


def get_perks_agent_instructions() -> str:
    """Return the system instructions for the Perks Agent."""
    return PERKS_AGENT_INSTRUCTIONS
