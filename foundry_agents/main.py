#!/usr/bin/env python
"""
Main entry point for the Foundry IQ + Agent Framework demo.

This demo showcases how to:
1. Create agents grounded to Foundry IQ Knowledge Bases
2. Orchestrate multiple agents using Microsoft Agent Framework
3. Route queries to the appropriate specialist agent

Usage:
    python main.py                    # Interactive mode
    python main.py --query "question" # Single query mode
    python main.py --demo             # Run demo queries
"""

import argparse
import asyncio
import sys

from config import load_config
from orchestrator.workflow import interactive_session, run_hr_assistant


DEMO_QUERIES = [
    "What's the difference between Northwind Health Plus and Standard plans?",
    "What is the company's policy on remote work?",
    "What wellness benefits does the company offer?",
    "How much vacation time do senior employees get?",
    "What's the deductible for the health insurance?",
]


async def run_demo(kb_variant: str = "base"):
    """Run a demonstration with sample queries."""
    config = load_config()
    
    print("\n" + "=" * 60)
    print("🎯 Foundry IQ + Agent Framework Demo")
    print("=" * 60)
    print(f"\nUsing Knowledge Base variant: {kb_variant}")
    print("This demo shows multi-agent orchestration with Foundry IQ")
    print("knowledge retrieval. Each query is routed to the appropriate")
    print("specialist agent based on its content.\n")
    
    for i, query in enumerate(DEMO_QUERIES, 1):
        print(f"\n{'─' * 60}")
        print(f"📝 Demo Query {i}/{len(DEMO_QUERIES)}:")
        print(f"   \"{query}\"")
        print("─" * 60)
        
        try:
            response = await run_hr_assistant(query, config, kb_variant=kb_variant)
            print(f"\n💬 Response:\n{response}")
        except Exception as e:
            print(f"\n❌ Error: {e}")
        
        if i < len(DEMO_QUERIES):
            print("\n⏳ Waiting 2 seconds before next query...")
            await asyncio.sleep(2)
    
    print("\n" + "=" * 60)
    print("✅ Demo complete!")
    print("=" * 60)


async def run_single_query(query: str, kb_variant: str = "base"):
    """Run a single query and print the response."""
    config = load_config()
    
    print(f"\n📝 Query: {query}")
    print(f"📚 Using KB: {kb_variant}\n")
    print("⏳ Processing...")
    
    try:
        response = await run_hr_assistant(query, config, kb_variant=kb_variant)
        print(f"\n💬 Response:\n{response}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


def validate_config():
    """Validate that required configuration is present."""
    config = load_config()
    
    errors = []
    
    if not config.search_endpoint:
        errors.append("AZURE_SEARCH_ENDPOINT is not set")
    if not config.openai_endpoint:
        errors.append("AZURE_OPENAI_ENDPOINT is not set")
    if not config.foundry_project_endpoint:
        errors.append("FOUNDRY_PROJECT_ENDPOINT is not set")
    
    if errors:
        print("❌ Configuration errors:")
        for error in errors:
            print(f"   • {error}")
        print("\nPlease copy .env.sample to .env and configure your settings.")
        print("See README.md for setup instructions.")
        sys.exit(1)
    
    return config


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Foundry IQ + Agent Framework Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                              # Interactive session
  python main.py --demo                       # Run demo queries
  python main.py -q "What are my benefits?"   # Single query
        """,
    )
    
    parser.add_argument(
        "-q", "--query",
        type=str,
        help="Run a single query and exit",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demonstration with sample queries",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate configuration and exit",
    )
    parser.add_argument(
        "--kb",
        type=str,
        choices=["base", "with-sharepoint", "with-web", "with-web-and-sharepoint"],
        default="base",
        help="Which Knowledge Base variant to use",
    )
    
    args = parser.parse_args()
    
    # Validate configuration
    config = validate_config()
    
    if args.validate:
        print("✅ Configuration is valid!")
        print(f"   Search Endpoint: {config.search_endpoint}")
        print(f"   OpenAI Endpoint: {config.openai_endpoint}")
        print(f"   Foundry Project: {config.foundry_project_endpoint}")
        print(f"   Retrieval Mode: {config.retrieval_mode}")
        sys.exit(0)
    
    # Run the appropriate mode
    if args.demo:
        asyncio.run(run_demo(kb_variant=args.kb))
    elif args.query:
        asyncio.run(run_single_query(args.query, kb_variant=args.kb))
    else:
        asyncio.run(interactive_session(config, kb_variant=args.kb))


if __name__ == "__main__":
    main()
