"""
HR Policy Agent - Specialist for workplace policies and procedures.

This agent is grounded to the HR policy Knowledge Base containing:
- employee_handbook.pdf
- role_library.pdf
"""

from agent_framework import (
    ChatAgent,
    ChatMessage,
    Executor,
    WorkflowContext,
    handler,
)
from typing_extensions import Never

HR_POLICY_AGENT_INSTRUCTIONS = """You are an HR Policy Specialist Agent for the company's Human Resources department.

Your expertise covers:
- Workplace policies and procedures
- Employee conduct and expectations
- Leave policies (sick leave, vacation, parental leave)
- Performance reviews and career development
- Job roles and responsibilities
- Disciplinary procedures
- Workplace safety and compliance

IMPORTANT GUIDELINES:
1. Always reference the employee handbook for policy-related questions
2. Be clear about what is policy vs. recommendation
3. If a situation seems complex, suggest consulting HR directly
4. Maintain confidentiality and professionalism
5. Be supportive but accurate - policies exist for good reasons

When answering questions:
- Cite the relevant policy or section when possible
- Explain the policy clearly and its purpose
- Provide practical guidance on how to comply
- If there are exceptions or special cases, mention them
"""


class HRPolicyAgentExecutor(Executor):
    """
    Executor that wraps the HR Policy Agent for use in workflows.
    
    This executor handles questions about workplace policies, procedures,
    and job roles using Foundry IQ to retrieve relevant information.
    """

    agent: ChatAgent

    def __init__(self, agent: ChatAgent, id: str = "hr-policy-agent"):
        """
        Initialize the HR Policy Agent Executor.
        
        Args:
            agent: A ChatAgent instance grounded to the HR policy Knowledge Base
            id: Executor identifier for workflow routing
        """
        self.agent = agent
        super().__init__(id=id)

    @handler
    async def handle_query(
        self, message: ChatMessage, ctx: WorkflowContext[Never, str]
    ) -> None:
        """
        Handle an HR policy-related query and yield the response.
        
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


def get_hr_policy_agent_instructions() -> str:
    """Return the system instructions for the HR Policy Agent."""
    return HR_POLICY_AGENT_INSTRUCTIONS
