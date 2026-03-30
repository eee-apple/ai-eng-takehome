#!/usr/bin/env python3
"""Interactive REPL for your agent framework.

This script provides a command-line interface to interact with your agent,
allowing you to enter prompts and receive streaming responses.
"""

import argparse

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from framework.agent import Agent, Tool
from framework.llm import OpenRouterConfig
from framework.stream_printer import StreamPrinter
from tools.database_tools import (
    DESCRIBE_COLUMN,
    EXECUTE_SQL,
    SAMPLE_VALUES,
)
from tools.discovery_tools import GET_FULL_SCHEMA, LIST_ALL_SCHEMAS, SAMPLE_TABLE, SUMMARIZE_TABLE
from tools.submit_answer import SUBMIT_ANSWER


def create_tools() -> dict[str, Tool]:
    """Create the tools for the agent.

    Returns:
        Dictionary mapping tool names to Tool instances.
    """
    return {
        SUBMIT_ANSWER.name: SUBMIT_ANSWER,
        DESCRIBE_COLUMN.name: DESCRIBE_COLUMN,
        SAMPLE_VALUES.name: SAMPLE_VALUES,
        EXECUTE_SQL.name: EXECUTE_SQL,
        LIST_ALL_SCHEMAS.name: LIST_ALL_SCHEMAS,
        GET_FULL_SCHEMA.name: GET_FULL_SCHEMA,
        SAMPLE_TABLE.name: SAMPLE_TABLE,
        SUMMARIZE_TABLE.name: SUMMARIZE_TABLE,
    }


def create_agent(api_key: str) -> Agent:
    """Create and configure the agent with default settings.

    Args:
        api_key: OpenRouter API key.

    Returns:
        Configured Agent instance.
    """
    config = OpenRouterConfig(
        api_key=api_key,
        compress_context=True,
        compress_keep_recent=5,
        compress_max_chars=300,
    )
    tools = create_tools()
    return Agent(config=config, tools=tools)


def print_welcome(console: Console) -> None:
    """Print a welcome message and usage instructions."""
    welcome_text = (
        "[bold cyan]Agent Interactive REPL[/bold cyan]\n\n"
        "Enter your prompts to interact with the agent.\n"
        "The agent has access to tools for answering questions.\n\n"
        "[dim]Commands:[/dim]\n"
        "  [yellow]quit[/yellow] or [yellow]exit[/yellow] - Exit the REPL\n"
        "  [yellow]reset[/yellow] - Reset the conversation history\n"
        "  [yellow]help[/yellow] - Show this help message"
    )
    console.print(Panel(welcome_text, title="Welcome", border_style="blue"))


def print_help(console: Console) -> None:
    """Print help information."""
    help_text = (
        "[bold]Available Commands:[/bold]\n\n"
        "  [yellow]quit[/yellow] / [yellow]exit[/yellow] - Exit the interactive session\n"
        "  [yellow]reset[/yellow] - Clear conversation history and start fresh\n"
        "  [yellow]help[/yellow] - Display this help message\n\n"
        "[bold]Tips:[/bold]\n"
        "  - Multi-line input is not supported; keep prompts on a single line"
    )
    console.print(Panel(help_text, title="Help", border_style="green"))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Interactive REPL for the agent framework"
    )
    parser.add_argument(
        "--api-key",
        required=True,
        help="OpenRouter API key",
    )
    return parser.parse_args()


def main() -> None:
    """Run the interactive REPL."""
    args = parse_args()

    console = Console()
    printer = StreamPrinter(
        show_thinking=True,
        show_tool_calls=True,
        show_tool_results=True,
        console=console,
    )

    print_welcome(console)

    console.print("\n[dim]Connecting to OpenRouter...[/dim]")
    agent = create_agent(args.api_key)
    console.print("[green]Connected successfully![/green]\n")

    while True:
        try:
            # Get user input
            user_input = Prompt.ask("[bold blue]You[/bold blue]")

            # Handle empty input
            if not user_input.strip():
                continue

            # Handle special commands
            command = user_input.strip().lower()
            if command in ("quit", "exit"):
                console.print("\n[dim]Goodbye![/dim]")
                break
            elif command == "reset":
                agent.reset_conversation()
                console.print("[yellow]Conversation reset.[/yellow]\n")
                continue
            elif command == "help":
                print_help(console)
                continue

            # Run the agent on the user input and stream the response
            console.print()
            events = agent.run(user_input)
            printer.print_stream(events)
            console.print()

        except KeyboardInterrupt:
            console.print("\n\n[dim]Interrupted. Type 'quit' to exit.[/dim]\n")
            continue
        except EOFError:
            console.print("\n[dim]Goodbye![/dim]")
            break


if __name__ == "__main__":
    main()
