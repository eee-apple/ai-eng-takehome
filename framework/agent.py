"""Agent framework for autonomous task execution with tool calling.

This module implements an agent that uses the OpenRouter API for LLM inference,
supporting streaming responses, tool calling, and reasoning token display.
"""

import json
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import duckdb

from framework.database import DATABASE_PATH
from framework.llm import OpenRouterClient, OpenRouterConfig, TokenUsage

# Prefix that indicates the agent should stop (answer was submitted)
# This avoids global state - the tool result signals completion
ANSWER_SUBMITTED_PREFIX = "ANSWER_SUBMITTED:"

type ToolFunction = Callable[..., str]


class EventType(Enum):
    """Types of events emitted during agent execution."""

    # Generation events
    GENERATION_START = auto()
    THINKING_START = auto()
    THINKING_CHUNK = auto()
    THINKING_END = auto()
    RESPONSE_CHUNK = auto()
    GENERATION_END = auto()

    # Tool events
    TOOL_CALL_START = auto()
    TOOL_CALL_PARSED = auto()
    TOOL_EXECUTION_START = auto()
    TOOL_EXECUTION_END = auto()

    # Agent loop events
    ITERATION_START = auto()
    ITERATION_END = auto()
    AGENT_COMPLETE = auto()
    AGENT_ERROR = auto()


@dataclass
class AgentEvent:
    """An event emitted during agent execution."""

    type: EventType
    data: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        """Format event for display."""
        return f"[{self.type.name}] {self.data}"


@dataclass
class Tool:
    """Represents a tool that can be called by the agent.

    Tool functions must return a string that will be shown to the LLM.
    """

    name: str
    description: str
    parameters: dict[str, Any]
    function: ToolFunction


@dataclass
class ToolCall:
    """Represents a (parsed) tool call request from the agent."""

    id: str  # Required for OpenAI-compatible API
    name: str
    arguments: dict[str, Any]
    error: str | None = None


@dataclass
class Message:
    """Represents a message in the conversation."""

    role: str  # "system", "user", "assistant", or "tool"
    content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None  # For assistant messages with tool calls
    tool_call_id: str | None = None  # For tool result messages


@dataclass
class ContextCompressionSettings:
    """Settings for context compression to reduce token usage."""

    enabled: bool = False
    keep_recent: int = 3  # Number of recent tool results to keep in full
    max_chars: int = 150  # Max chars for truncated older results


@dataclass
class Conversation:
    """Represents a conversation between the agent and the user."""

    messages: list[Message] = field(default_factory=list)

    def to_api_format(
        self,
        compression: ContextCompressionSettings | None = None,
    ) -> list[dict[str, Any]]:
        """Convert the conversation to OpenAI-compatible API format.

        Args:
            compression: Optional compression settings. If enabled, older tool
                results are truncated and duplicate consecutive tool calls are
                deduplicated.
        """
        messages_to_convert = self.messages

        if compression and compression.enabled:
            messages_to_convert = _compress_messages(
                self.messages,
                keep_recent=compression.keep_recent,
                max_chars=compression.max_chars,
            )

        result: list[dict[str, Any]] = []
        for message in messages_to_convert:
            msg: dict[str, Any] = {"role": message.role}

            if message.content is not None:
                msg["content"] = message.content

            if message.tool_calls is not None:
                msg["tool_calls"] = message.tool_calls

            if message.tool_call_id is not None:
                msg["tool_call_id"] = message.tool_call_id

            result.append(msg)
        return result


def _truncate_tool_result(content: str, max_chars: int) -> str:
    """Truncate a tool result to max_chars with a summary prefix."""
    if len(content) <= max_chars:
        return content

    # Extract first line as summary (often contains row/column counts)
    first_line = content.split("\n")[0]
    if len(first_line) <= max_chars - 20:
        return f"[Truncated] {first_line}"

    return f"[Truncated] {content[:max_chars - 15]}..."


def _compress_messages(
    messages: list[Message],
    keep_recent: int,
    max_chars: int,
) -> list[Message]:
    """Compress messages by truncating old tool results and deduplicating.

    Applies two optimizations:
    1. Truncates tool results older than keep_recent to max_chars
    2. Removes duplicate consecutive tool calls with identical results
    """
    # Find all tool message indices (for determining which are "recent")
    tool_indices: list[int] = [
        i for i, m in enumerate(messages) if m.role == "tool"
    ]

    # Indices of tool messages to keep in full (the most recent ones)
    recent_tool_indices = set(tool_indices[-keep_recent:]) if tool_indices else set()

    # Build compressed message list
    result: list[Message] = []
    seen_tool_calls: dict[str, str] = {}  # (name, args_json) -> full result

    for i, msg in enumerate(messages):
        if msg.role == "tool":
            # Check for deduplication: same tool call with same result
            # Find the corresponding assistant message's tool call
            tool_key: str | None = None
            for j in range(i - 1, -1, -1):
                assistant_tool_calls = messages[j].tool_calls
                if messages[j].role == "assistant" and assistant_tool_calls:
                    for tc in assistant_tool_calls:
                        if tc.get("id") == msg.tool_call_id:
                            name = tc.get("function", {}).get("name", "")
                            args = tc.get("function", {}).get("arguments", "")
                            tool_key = f"{name}:{args}"
                            break
                    break

            # Deduplicate: if we've seen this exact call before with same result
            if tool_key and msg.content:
                if tool_key in seen_tool_calls:
                    prev_content = seen_tool_calls[tool_key]
                    if prev_content == msg.content:
                        # Skip this duplicate - but we need to keep the message
                        # structure for the API, so mark it as deduplicated
                        result.append(
                            Message(
                                role=msg.role,
                                content="[Duplicate call - see earlier result]",
                                tool_call_id=msg.tool_call_id,
                            )
                        )
                        continue
                seen_tool_calls[tool_key] = msg.content

            # Truncate if not in recent set
            if i not in recent_tool_indices and msg.content:
                result.append(
                    Message(
                        role=msg.role,
                        content=_truncate_tool_result(msg.content, max_chars),
                        tool_call_id=msg.tool_call_id,
                    )
                )
            else:
                result.append(msg)
        else:
            result.append(msg)

    return result


def _summarize_tables(
    schema_name: str,
    table_names: list[str],
) -> str:
    """Run SUMMARIZE on tables and return full output for context.

    Returns all column stats from DuckDB's SUMMARIZE command, trusting
    the model to use what it needs.
    """
    parts: list[str] = []
    conn = None
    try:
        conn = duckdb.connect(str(DATABASE_PATH), read_only=True)
        for table in table_names:
            try:
                df = conn.execute(
                    f'SUMMARIZE "{schema_name}"."{table}"'
                ).fetchdf()
                if df.empty:
                    continue

                lines: list[str] = []
                lines.append(f"### {schema_name}.{table}")
                lines.append(
                    "| Column | Type | Min | Max | Distinct | Null% |"
                )
                lines.append("| --- | --- | --- | --- | --- | --- |")
                for _, row in df.iterrows():
                    col_name = str(row.get("column_name", ""))
                    col_type = str(row.get("column_type", ""))
                    min_val = str(row.get("min", ""))[:40]
                    max_val = str(row.get("max", ""))[:40]
                    distinct = int(row.get("approx_unique", 0))
                    null_pct = row.get("null_percentage", 0)
                    lines.append(
                        f"| {col_name} | {col_type} | {min_val} "
                        f"| {max_val} | {distinct} "
                        f"| {null_pct:.0f}% |"
                    )
                parts.append("\n".join(lines))
            except Exception:
                continue
    except Exception:
        pass
    finally:
        if conn is not None:
            conn.close()

    if not parts:
        return ""
    return "## Table Summaries (auto-generated)\n\n" + "\n\n".join(parts)


def _extract_tables_from_plan(
    plan_text: str, schema_name: str,
) -> list[str]:
    """Extract table names from the plan text.

    Looks for schema.table patterns and bare table names after keywords.
    """
    tables: list[str] = []
    seen: set[str] = set()

    # Match schema.table patterns
    for m in re.finditer(
        rf"\b{re.escape(schema_name)}\.(\w+)\b", plan_text, re.IGNORECASE,
    ):
        t = m.group(1)
        if t.lower() not in seen:
            seen.add(t.lower())
            tables.append(t)

    # Also match "Tables to query" style lists — lines with table names
    # after common headers
    table_section = re.search(
        r"[Tt]ables?\s*(?:to query|needed|:)\s*[:\-]?\s*(.*?)(?:\n\s*\n|\n-\s*\*\*|$)",
        plan_text,
        re.DOTALL,
    )
    if table_section:
        for word in re.findall(r"`(\w+)`|\b([A-Z]\w+)\b", table_section.group(1)):
            t = word[0] or word[1]
            if t.lower() not in seen and t.lower() != schema_name.lower():
                seen.add(t.lower())
                tables.append(t)

    return tables


@dataclass
class TaggedRule:
    """A business rule tagged as FILTER or TRANSFORM."""

    text: str
    tag: str  # "FILTER" or "TRANSFORM"


# Keywords that indicate transform/rename/merge rules
_TRANSFORM_KEYWORDS = re.compile(
    r"\b(aggregate|merge|combin|rebrand|rename|reclassif|"
    r"map\w*\s+to|categoriz|classif|taxonomy|"
    r"segment\s+separately|flag\s+and\s+analyz|"
    r"analyz\w+\s+separately|attribute\s+to|"
    r"catch-all|use\s+\S+\s+as\s+a)\b",
    re.IGNORECASE,
)


def _classify_rule(rule_text: str) -> str:
    """Classify a rule as FILTER or TRANSFORM based on keywords."""
    if _TRANSFORM_KEYWORDS.search(rule_text):
        return "TRANSFORM"
    return "FILTER"


def _parse_business_rules(context: str) -> list[TaggedRule]:
    """Extract individual business rules from the context, tagged by type.

    FILTER rules: exclusions, inclusions, conditions on data.
    TRANSFORM rules: renaming, merging, reclassifying, aggregating.
    """
    rules: list[TaggedRule] = []

    # Find the business rules section
    marker = "## ⚠️ BUSINESS RULES (MUST READ AND APPLY)"
    idx = context.find(marker)
    if idx < 0:
        return rules

    rules_text = context[idx + len(marker):]

    # Stop at the next major section or end
    end_markers = ["## Schema Context", "## Table Summaries", "## Planning Summary"]
    for em in end_markers:
        end_idx = rules_text.find(em)
        if end_idx >= 0:
            rules_text = rules_text[:end_idx]

    # Extract bullet points (lines starting with -)
    current_section = ""
    for line in rules_text.split("\n"):
        line = line.strip()
        if line.startswith("## ") or line.startswith("# "):
            current_section = line.lstrip("#").strip()
        elif line.startswith("- ") and len(line) > 10:
            rule_text = line[2:].strip()
            if current_section:
                full_text = f"[{current_section}] {rule_text}"
            else:
                full_text = rule_text
            tag = _classify_rule(rule_text)
            rules.append(TaggedRule(text=full_text, tag=tag))

    return rules


def _run_sanity_checks(query: str, result_text: str) -> str:
    """Run mechanical sanity checks on a SQL query and its results.

    Returns a string of warnings (empty if none).
    """
    warnings: list[str] = []
    query_upper = query.upper()

    # Check 1: JOINed tables not used in SELECT
    # Extract table aliases from JOINs
    join_aliases: dict[str, str] = {}  # alias -> table
    for m in re.finditer(
        r"(?:JOIN|FROM)\s+(\w+\.\w+)(?:\s+(?:AS\s+)?(\w+))?",
        query, re.IGNORECASE,
    ):
        table = m.group(1)
        alias = m.group(2) or table
        join_aliases[alias.upper()] = table

    # Extract what's referenced in SELECT
    select_match = re.search(r"SELECT\s+(.*?)FROM", query_upper, re.DOTALL)
    if select_match and len(join_aliases) > 1:
        select_clause = select_match.group(1)
        for alias, table in join_aliases.items():
            # Check if alias appears in SELECT clause
            if alias + "." not in select_clause and alias not in select_clause:
                # Check original case too
                alias_lower = alias.lower()
                select_lower = select_match.group(1).lower() if select_match else ""
                query_select = re.search(r"SELECT\s+(.*?)FROM", query, re.DOTALL | re.IGNORECASE)
                if query_select:
                    sel = query_select.group(1)
                    # Check if any variant of the alias appears
                    found = False
                    for a_variant in [alias, alias.lower(), alias.capitalize()]:
                        if a_variant + "." in sel or f" {a_variant}" in sel:
                            found = True
                            break
                    if not found:
                        warnings.append(
                            f"⚠️ Table {table} (alias {alias}) is JOINed but "
                            f"none of its columns appear in SELECT. This may "
                            f"be an unnecessary lookup JOIN."
                        )

    # Check 2: Cancelled filter missing for airline-like queries
    if "CANCELLED" in query_upper or "CANCEL" in query_upper:
        pass  # has it
    elif any(kw in query_upper for kw in ["DEPTIME", "ARRTIME", "DELAY", "FLIGHT"]):
        if "CANCELLED" not in query_upper:
            warnings.append(
                "⚠️ Query references flight data but has no Cancelled filter. "
                "Should cancelled flights be excluded?"
            )

    # Check 3: Result magnitude check for aggregations
    if result_text and ("1 rows, 1 cols" in result_text or "1 rows," in result_text):
        # Single-row result — check if it's a suspiciously large number
        numbers = re.findall(r"\b(\d{6,})\b", result_text)
        if numbers and ("COUNT" in query_upper or "SUM" in query_upper):
            warnings.append(
                f"⚠️ Single-row result with large value(s): {numbers[:3]}. "
                f"Verify: did the question ask for COUNT (number of items) "
                f"or SUM (total amount)?"
            )

    if not warnings:
        return ""
    return "## ⚠️ Automated Sanity Checks\n" + "\n".join(warnings)


def _compare_candidates(
    a: dict[str, str], b: dict[str, str],
) -> str:
    """Build a structured comparison between two candidate queries."""
    diffs: list[str] = []

    # Row count comparison
    def _extract_row_count(result: str) -> int | None:
        m = re.search(r"(\d+)\s+rows", result)
        return int(m.group(1)) if m else None

    rows_a = _extract_row_count(a["result"])
    rows_b = _extract_row_count(b["result"])
    if rows_a is not None and rows_b is not None and rows_a != rows_b:
        diffs.append(
            f"- **Row count differs**: A returns {rows_a} rows, "
            f"B returns {rows_b} rows. If B has far fewer rows, it may "
            f"be over-filtering with TRANSFORM rules the question didn't ask for."
        )

    # JOIN count comparison
    def _count_joins(query: str) -> int:
        return len(re.findall(r"\bJOIN\b", query, re.IGNORECASE))

    joins_a = _count_joins(a["query"])
    joins_b = _count_joins(b["query"])
    if joins_a != joins_b:
        diffs.append(
            f"- **JOIN count differs**: A has {joins_a} JOINs, "
            f"B has {joins_b}. Extra JOINs may be unnecessary lookups."
        )

    # WHERE clause count
    def _count_filters(query: str) -> int:
        return len(re.findall(r"\bAND\b|\bOR\b", query, re.IGNORECASE))

    filters_a = _count_filters(a["query"])
    filters_b = _count_filters(b["query"])
    if filters_a != filters_b:
        diffs.append(
            f"- **Filter count differs**: A has ~{filters_a} conditions, "
            f"B has ~{filters_b}. More filters may mean over-filtering."
        )

    if not diffs:
        return ""
    return (
        "## ⚠️ Candidate Comparison\n"
        + "\n".join(diffs)
        + "\n\n"
    )


def _parse_schema_names(schema_directory: str) -> list[str]:
    """Extract schema names from the list_all_schemas output.

    Each schema line looks like: "SchemaName [N tables]: table1(100), ..."
    """
    names: list[str] = []
    for line in schema_directory.split("\n"):
        m = re.match(r"^(\w+)\s+\[", line)
        if m:
            names.append(m.group(1))
    return names


def _extract_schema_from_plan(plan_text: str, known_schemas: list[str] | None = None) -> str:
    """Extract the chosen schema name from plan text.

    If known_schemas is provided, validates against the actual schema list.
    """
    # Try "**Schema chosen**: X" or "Schema: X" patterns
    m = re.search(r"[Ss]chema\s*(?:chosen)?[*:\s`]+(\w+)", plan_text)
    if m:
        candidate = m.group(1)
        if known_schemas:
            # Validate against known schemas (case-insensitive)
            for s in known_schemas:
                if s.lower() == candidate.lower():
                    return s
        return candidate

    # Try schema.table pattern — take the schema part
    if known_schemas:
        lower_schemas = {s.lower(): s for s in known_schemas}
        for m in re.finditer(r"\b(\w+)\.\w+\b", plan_text):
            candidate = m.group(1).lower()
            if candidate in lower_schemas:
                return lower_schemas[candidate]

    return ""


class Agent:
    """Implements a tiny, generic agent framework.

    Built on top of the OpenRouter API client.

    Only supports a single model, streaming, and an extensible tool set.
    """

    def __init__(self, config: OpenRouterConfig, tools: dict[str, Tool]):
        self.config = config
        self.tools: dict[str, Tool] = tools  # mapping from tool name to tool object
        self.client: OpenRouterClient = OpenRouterClient(config)
        self.conversation: Conversation = Conversation()
        self._compression = ContextCompressionSettings(
            enabled=config.compress_context,
            keep_recent=config.compress_keep_recent,
            max_chars=config.compress_max_chars,
        )
        self.reset_conversation()

    def _get_tool_definitions(self) -> list[dict[str, Any]]:
        """Get tool definitions in OpenAI-compatible format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in self.tools.values()
        ]

    def _execute_tool(self, tool_call: ToolCall) -> str:
        """Execute a tool call and return the result as a string.

        Guaranteed to return a string, swallowing exceptions.
        """
        if tool_call.error:
            return f"Error parsing arguments for tool '{tool_call.name}': {tool_call.error}"

        if tool_call.name not in self.tools:
            return f"Error: Unknown tool '{tool_call.name}'"
        tool = self.tools[tool_call.name]
        try:
            return tool.function(**tool_call.arguments)
        except Exception as e:
            return f"Error executing {tool_call.name}: {e}"

    def _generate_response(self, conversation: Conversation) -> Iterator[AgentEvent]:
        """Generate a response from the model, streaming the events out."""
        yield AgentEvent(type=EventType.GENERATION_START)

        messages = conversation.to_api_format(compression=self._compression)
        tools = self._get_tool_definitions() if self.tools else None

        full_content = ""
        tool_calls: list[dict[str, Any]] = []
        in_thinking = False
        finish_reason: str | None = None
        usage: TokenUsage | None = None

        for chunk in self.client.chat_completion_stream(messages, tools):
            # Handle reasoning/thinking tokens
            if chunk.reasoning_details:
                for detail in chunk.reasoning_details:
                    if detail.get("type") == "reasoning.text":
                        text = detail.get("text", "")
                        if text:
                            if not in_thinking:
                                in_thinking = True
                                yield AgentEvent(type=EventType.THINKING_START)
                            yield AgentEvent(
                                type=EventType.THINKING_CHUNK,
                                data={"chunk": text},
                            )

            # Handle regular content
            if chunk.content:
                # Close thinking block if we were in it
                if in_thinking:
                    in_thinking = False
                    yield AgentEvent(type=EventType.THINKING_END)

                full_content += chunk.content
                yield AgentEvent(
                    type=EventType.RESPONSE_CHUNK,
                    data={"chunk": chunk.content},
                )

            # Handle tool calls (accumulated at the end)
            if chunk.tool_calls:
                tool_calls = chunk.tool_calls

            if chunk.finish_reason:
                finish_reason = chunk.finish_reason

            # Capture usage data (comes in final chunk)
            if chunk.usage:
                usage = chunk.usage

        # Close thinking if still open
        if in_thinking:
            yield AgentEvent(type=EventType.THINKING_END)

        event_data: dict[str, Any] = {
            "full_response": full_content,
            "tool_calls": tool_calls,
            "finish_reason": finish_reason,
        }
        if usage:
            event_data["usage"] = usage

        yield AgentEvent(type=EventType.GENERATION_END, data=event_data)

    def _run_phase(
        self,
        conversation: Conversation,
        tools: dict[str, Tool],
        max_iterations: int,
        phase_name: str,
    ) -> Iterator[AgentEvent | str]:
        """Run a conversation phase with given tools and budget.

        Yields AgentEvents for streaming, and returns the final text as a str.
        Temporarily swaps self.tools and self.conversation for the phase.
        """
        original_tools = self.tools
        original_conversation = self.conversation
        self.tools = tools
        self.conversation = conversation
        final_text = ""

        try:
            total_usage = TokenUsage()

            for iteration in range(max_iterations):
                yield AgentEvent(
                    type=EventType.ITERATION_START,
                    data={"iteration": iteration + 1, "phase": phase_name},
                )

                full_response = ""
                tool_calls_data: list[dict[str, Any]] = []

                for event in self._generate_response(conversation):
                    yield event
                    if event.type == EventType.GENERATION_END:
                        full_response = event.data.get("full_response", "")
                        tool_calls_data = event.data.get("tool_calls", [])
                        if "usage" in event.data and event.data["usage"]:
                            total_usage = total_usage + event.data["usage"]

                tool_calls = _parse_tool_calls_from_api(tool_calls_data)

                if not tool_calls:
                    is_empty = not full_response or not full_response.strip()
                    looks_like_failed = full_response and "{" in full_response

                    if is_empty or looks_like_failed:
                        conversation.messages.append(
                            Message(role="assistant", content=full_response or "")
                        )
                        conversation.messages.append(
                            Message(
                                role="user",
                                content=(
                                    "You must use the submit_answer TOOL to submit "
                                    "your answer. Call submit_answer now."
                                ),
                            )
                        )
                        continue

                    conversation.messages.append(
                        Message(role="assistant", content=full_response)
                    )
                    final_text = full_response
                    yield AgentEvent(
                        type=EventType.AGENT_COMPLETE,
                        data={"response": full_response, "usage": total_usage,
                              "phase": phase_name},
                    )
                    yield final_text
                    return

                yield AgentEvent(
                    type=EventType.TOOL_CALL_START,
                    data={"count": len(tool_calls)},
                )

                conversation.messages.append(
                    Message(
                        role="assistant",
                        content=full_response if full_response else None,
                        tool_calls=tool_calls_data,
                    )
                )

                for tool_call in tool_calls:
                    yield AgentEvent(
                        type=EventType.TOOL_CALL_PARSED,
                        data={"name": tool_call.name, "arguments": tool_call.arguments},
                    )
                    yield AgentEvent(
                        type=EventType.TOOL_EXECUTION_START,
                        data={"name": tool_call.name},
                    )
                    tool_result = self._execute_tool(tool_call)
                    yield AgentEvent(
                        type=EventType.TOOL_EXECUTION_END,
                        data={"name": tool_call.name, "result": tool_result},
                    )

                    conversation.messages.append(
                        Message(
                            role="tool",
                            content=tool_result,
                            tool_call_id=tool_call.id,
                        )
                    )

                    if tool_result.startswith(ANSWER_SUBMITTED_PREFIX):
                        yield AgentEvent(
                            type=EventType.AGENT_COMPLETE,
                            data={
                                "reason": "answer_submitted",
                                "tool": tool_call.name,
                                "usage": total_usage,
                                "phase": phase_name,
                            },
                        )
                        yield final_text
                        return

                yield AgentEvent(
                    type=EventType.ITERATION_END,
                    data={"iteration": iteration + 1, "phase": phase_name},
                )

            yield AgentEvent(
                type=EventType.AGENT_ERROR,
                data={"error": f"Max iterations ({phase_name})",
                      "usage": total_usage, "phase": phase_name},
            )
            yield final_text
        finally:
            self.tools = original_tools
            self.conversation = original_conversation

    # ── Planning Phase ──────────────────────────────────────────────────
    _PLAN_SYSTEM_PROMPT = (
        "You are a database planning agent. Given a question and a list of "
        "available schemas, identify the right schema and produce a plan.\n\n"
        "## Workflow\n"
        "You MUST use your tools to explore the database before writing a plan.\n"
        "1. Review the Available Schemas list (already provided). Pick the "
        "best-fit schema based on table names and the question.\n"
        "2. Call get_full_schema on your chosen schema (and the runner-up "
        "if uncertain).\n"
        "3. Call summarize_table on each table you plan to query — this "
        "shows column types, value ranges, and nulls.\n"
        "4. Write a plan with:\n"
        "   - **Schema chosen** and why.\n"
        "   - **Tables to query** — only tables you need columns from. "
        "Skip lookup/reference tables unless the question asks for names "
        "or descriptions.\n"
        "   - **Key columns** with data ranges from summarize_table.\n"
        "   - **Data quirks** (nulls, unexpected ranges, date boundaries).\n\n"
        "## RULE ANALYSIS (required if business rules exist)\n"
        "For EACH business rule in the schema, write:\n"
        "- **Rule**: <quote the rule verbatim>\n"
        "- **Applies**: YES or NO, with a brief reason tied to the question\n"
        "- **SQL impact**: what WHERE/JOIN/CASE clause this requires (if YES)\n\n"
        "Guidance:\n"
        "- Filtering/exclusion rules (exclude X, only count Y) → YES if the "
        "query touches that data domain.\n"
        "- Rename/merge/reclassify rules (aggregate as X, map to Y) → YES "
        "only if the question explicitly mentions those categories.\n"
    )

    # ── Draft Phase ──────────────────────────────────────────────────
    _DRAFT_SYSTEM_PROMPT = (
        "You are an expert SQL agent working with a DuckDB database.\n"
        "A planning phase has identified the schema, and a schema linking "
        "phase has mapped question concepts to specific columns.\n\n"
        "## Workflow\n"
        "You MUST use tools — do not just write text.\n"
        "1. Read the Schema Linking section — you MUST use the mapped "
        "columns and tables. Do not substitute alternatives.\n"
        "2. Decide which business rules apply to THIS question.\n"
        "3. Write a SQL query answering the question.\n"
        "4. Call execute_sql to run and verify your query.\n"
        "5. Call submit_answer with the final query when confident.\n\n"
        "## Business Rules\n"
        "The context includes business rules for this domain. For each "
        "rule, decide if it applies to the question:\n"
        "- Filtering/exclusion rules → apply if your query touches that "
        "data (e.g., a rule about weather delays applies to any query on "
        "flight performance).\n"
        "- Rename/merge/reclassify rules → apply ONLY if the question "
        "explicitly mentions those categories.\n\n"
        "## SQL Rules\n"
        "- ALWAYS use schema.table syntax (e.g., schema.table_name).\n"
        "- Do NOT add filters beyond what the question and applicable "
        "business rules require. Simpler queries are usually more correct.\n"
        "- Include ALL relevant columns — extra columns are fine, "
        "missing columns cause failures.\n"
        "- Only JOIN when you need columns from another table. Never join "
        "lookup tables just to resolve codes into names unless the question "
        "explicitly asks for names/descriptions.\n"
        "- Keep first and last names as SEPARATE columns.\n"
        "- Return rates as decimals (0.0-1.0) unless the question says "
        "'percentage' or '%'.\n"
    )

    # ── Review Phase ─────────────────────────────────────────────────
    _REVIEW_SYSTEM_PROMPT = (
        "You are a SQL review agent. You receive candidate queries and "
        "must pick the best one, fix it if needed, and submit.\n\n"
        "You MUST use tools — do not just write text.\n\n"
        "## Workflow\n"
        "1. Compare the candidates. Read any sanity-check warnings.\n"
        "2. Audit each numbered business rule against the best candidate.\n"
        "3. Fix the query if needed, call execute_sql to test.\n"
        "4. Call submit_answer with the final query.\n\n"
        "## Candidate Selection\n"
        "- Candidate A is minimal (may be missing rules).\n"
        "- Candidate B follows rules more aggressively (may over-apply).\n"
        "- Check the Schema Linking section — the query MUST use the "
        "mapped columns and tables, not alternatives.\n"
        "- Pick the one closer to correct, then fix what's wrong.\n\n"
        "## Rule Audit\n"
        "For each numbered rule, decide APPLY or SKIP:\n"
        "- Filtering/exclusion rules → APPLY if the query touches that "
        "data.\n"
        "- Rename/merge/reclassify rules → APPLY only if the question "
        "explicitly mentions those categories.\n"
        "- If APPLY: is it correctly implemented? If missing, fix it.\n"
        "- If SKIP: is it accidentally in the query? If so, REMOVE it.\n\n"
        "## Clause Audit\n"
        "For EVERY WHERE, CASE, HAVING, GROUP BY, and JOIN clause:\n"
        "- Is it required by the question text? → KEEP\n"
        "- Is it implementing an APPLY rule? → KEEP\n"
        "- Otherwise → REMOVE. The #1 mistake is over-filtering.\n\n"
        "## Other Checks\n"
        "- No unnecessary lookup-table JOINs (don't join to resolve codes "
        "into names unless the question asks for names/descriptions).\n"
        "- First/last names are SEPARATE columns.\n"
        "- All relevant columns included.\n"
    )

    # ── Schema Linking Phase ─────────────────────────────────────────
    _LINK_SYSTEM_PROMPT = (
        "You are a schema linking agent. Map each concept in the question "
        "to specific schema elements.\n\n"
        "## Instructions\n"
        "Given a question, schema, and business rules, produce a mapping:\n\n"
        "For EACH concept/entity/metric in the question, write:\n"
        "- **Concept**: <phrase from question>\n"
        "- **Maps to**: <schema.table.column>\n"
        "- **Reason**: <why this column, not alternatives>\n\n"
        "Also produce:\n"
        "- **Tables needed**: list ONLY tables whose columns appear in "
        "your mappings. Do NOT include lookup/reference tables unless "
        "the question asks for names or descriptions.\n"
        "- **Applicable rules**: for each business rule, state if it "
        "applies and what SQL clause it requires.\n\n"
        "## Key Guidance\n"
        "- If the question says 'by X' or 'for each X', map X to a "
        "column to GROUP BY — use that column directly, do not JOIN "
        "a lookup table to resolve it.\n"
        "- 'total X' usually means COUNT, 'total amount' means SUM.\n"
        "- If a concept is ambiguous across schemas, explain the "
        "disambiguation and pick the best fit.\n"
    )

    def run(self, prompt: str) -> Iterator[AgentEvent]:
        """Run the agent with a plan→link→draft→review approach.

        Planning: Identifies schema, gathers tables/columns/rules/samples.
        Schema Linking: Maps question terms to specific columns/tables.
        Draft: Writes the SQL query using the gathered context.
        Review: Reviews draft query with fresh eyes, same context.
        """
        # ── Budget allocation ──
        plan_budget = 12
        link_budget = 3
        draft_budget = 28
        review_budget = self.config.max_iterations - plan_budget - link_budget - draft_budget

        # ── Pre-discover all schemas upfront ──
        schema_directory = ""
        if "list_all_schemas" in self.tools:
            schema_directory = self.tools["list_all_schemas"].function()

        # ── Planning Phase ──
        plan_tools = {}
        for name in ("get_full_schema", "sample_table",
                      "summarize_table"):
            if name in self.tools:
                plan_tools[name] = self.tools[name]

        plan_conversation = Conversation()
        plan_conversation.messages.append(
            Message(role="system", content=self._PLAN_SYSTEM_PROMPT)
        )
        plan_conversation.messages.append(
            Message(
                role="user",
                content=(
                    f"Question: {prompt}\n\n"
                    f"## Available Schemas\n{schema_directory}"
                ),
            )
        )

        # Collect planning context
        plan_context_parts: list[str] = []
        plan_text = ""

        for item in self._run_phase(
            plan_conversation, plan_tools, plan_budget, "plan"
        ):
            if isinstance(item, str):
                plan_text = item
            elif isinstance(item, AgentEvent):
                if (item.type == EventType.TOOL_EXECUTION_END
                        and item.data.get("result")):
                    tool_name = item.data.get("name", "")
                    if tool_name in ("get_full_schema", "sample_table",
                                      "summarize_table"):
                        plan_context_parts.append(item.data["result"])
                if item.type in (EventType.AGENT_COMPLETE, EventType.AGENT_ERROR):
                    pass
                else:
                    yield item

        # Extract known schema names from the directory we already fetched
        known_schemas = _parse_schema_names(schema_directory)

        # Fallback: if planner never called get_full_schema, call it now
        has_schema = any("## Schema:" in p or "### " in p for p in plan_context_parts)
        if not has_schema and "get_full_schema" in self.tools:
            schema_name = _extract_schema_from_plan(plan_text, known_schemas)
            if schema_name:
                try:
                    result = self.tools["get_full_schema"].function(
                        schema_name=schema_name
                    )
                    plan_context_parts.append(result)
                except Exception:
                    pass

        # Build context block — pass all rules directly to draft/review
        # (don't filter through planner's YES/NO — let draft decide)
        schema_context = "\n\n".join(plan_context_parts)

        # Auto-summarize tables if planner didn't already call summarize_table
        has_summarize = any("SUMMARIZE" in p for p in plan_context_parts)
        chosen_schema = _extract_schema_from_plan(plan_text, known_schemas)
        plan_tables = _extract_tables_from_plan(plan_text, chosen_schema) if chosen_schema else []
        table_summary = ""
        if not has_summarize and plan_tables:
            table_summary = _summarize_tables(chosen_schema, plan_tables)

        # Extract planner's rule analysis as advisory context
        planner_rule_analysis = ""
        if plan_text:
            ra_match = re.search(
                r"## RULE ANALYSIS\s*\n(.*?)(?=\n## |\Z)",
                plan_text,
                re.DOTALL,
            )
            if ra_match:
                planner_rule_analysis = ra_match.group(1).strip()

        gathered_parts = []
        if planner_rule_analysis:
            gathered_parts.append(
                "## ⚠️ Planner's Rule Recommendations (advisory)\n"
                "The planner analyzed each business rule for this question. "
                "Use these recommendations to guide your decisions.\n\n"
                + planner_rule_analysis
            )
        gathered_parts.append(f"## Schema Context\n{schema_context}")
        if table_summary:
            gathered_parts.append(table_summary)
        if plan_text:
            gathered_parts.append(f"## Planning Summary\n{plan_text}")
        gathered_context = "\n\n".join(gathered_parts)

        # ── Schema Linking Phase ──
        link_conversation = Conversation()
        link_conversation.messages.append(
            Message(role="system", content=self._LINK_SYSTEM_PROMPT)
        )
        link_conversation.messages.append(
            Message(
                role="user",
                content=(
                    f"## Question\n{prompt}\n\n"
                    f"{gathered_context}"
                ),
            )
        )

        schema_linking = ""
        for item in self._run_phase(
            link_conversation, {}, link_budget, "link"
        ):
            if isinstance(item, str):
                schema_linking = item
            elif isinstance(item, AgentEvent):
                if item.type in (EventType.AGENT_COMPLETE, EventType.AGENT_ERROR):
                    pass
                else:
                    yield item

        # Prepend schema linking to gathered context for draft/review
        if schema_linking:
            gathered_context = (
                f"## ⚠️ Schema Linking (use these mappings)\n"
                f"{schema_linking}\n\n"
                f"{gathered_context}"
            )

        # ── Draft Phase — generate two candidates ──
        draft_tools = {}
        for name in ("execute_sql", "submit_answer", "sample_table",
                      "sample_values", "describe_column", "get_full_schema",
                      "summarize_table"):
            if name in self.tools:
                draft_tools[name] = self.tools[name]

        # Pre-parse and tag business rules
        all_rules = _parse_business_rules(gathered_context)
        filter_rules = [r for r in all_rules if r.tag == "FILTER"]

        # Format rule blocks for each candidate
        def _format_rules(rules: list[TaggedRule], show_tags: bool = False) -> str:
            if not rules:
                return ""
            lines = []
            for i, r in enumerate(rules):
                tag_prefix = f"[{r.tag}] " if show_tags else ""
                lines.append(f"  {i+1}. {tag_prefix}{r.text}")
            return (
                "## Business Rules to Apply\n"
                + "\n".join(lines)
                + "\n\n"
            )

        filter_rules_block = _format_rules(filter_rules)
        all_rules_block = _format_rules(all_rules)

        candidates: list[dict[str, str]] = []  # [{query, result}, ...]
        half_budget = draft_budget // 2

        draft_configs = [
            # Candidate A: conservative — only sees FILTER rules
            (
                "Write the SIMPLEST possible query. Use only columns and "
                "tables directly needed. Do NOT join lookup/reference tables. "
                "Return raw codes/IDs rather than resolved names.",
                filter_rules_block,
            ),
            # Candidate B: thorough — sees all rules
            (
                "Write a thorough query. Apply all listed business rules "
                "that are relevant to this question's data domain.",
                all_rules_block,
            ),
        ]

        for draft_idx, (draft_instruction, rules_block) in enumerate(
            draft_configs
        ):
            draft_conversation = Conversation()
            draft_conversation.messages.append(
                Message(role="system", content=self._DRAFT_SYSTEM_PROMPT)
            )
            draft_conversation.messages.append(
                Message(
                    role="user",
                    content=(
                        f"## Question\n{prompt}\n\n"
                        f"## Approach\n{draft_instruction}\n\n"
                        f"{rules_block}"
                        f"{gathered_context}"
                    ),
                )
            )

            query = ""
            result = ""

            for item in self._run_phase(
                draft_conversation, draft_tools, half_budget,
                f"draft_{draft_idx + 1}",
            ):
                if isinstance(item, str):
                    pass
                elif isinstance(item, AgentEvent):
                    if (item.type == EventType.TOOL_EXECUTION_END
                            and item.data.get("name") == "submit_answer"
                            and item.data.get("result", "").startswith(ANSWER_SUBMITTED_PREFIX)):
                        query = item.data["result"][len(ANSWER_SUBMITTED_PREFIX):]
                    elif (item.type == EventType.TOOL_EXECUTION_END
                          and item.data.get("name") == "execute_sql"):
                        result = item.data.get("result", "")
                    if item.type in (EventType.AGENT_COMPLETE, EventType.AGENT_ERROR):
                        pass
                    else:
                        yield item

            # Extract query from conversation if not submitted
            if not query:
                for msg in reversed(draft_conversation.messages):
                    if msg.role == "tool" and msg.content and msg.tool_call_id:
                        for j in range(len(draft_conversation.messages) - 1, -1, -1):
                            assistant_msg = draft_conversation.messages[j]
                            if assistant_msg.role == "assistant" and assistant_msg.tool_calls:
                                for tc in assistant_msg.tool_calls:
                                    if (tc.get("id") == msg.tool_call_id
                                            and tc.get("function", {}).get("name") == "execute_sql"):
                                        args = json.loads(tc["function"].get("arguments", "{}"))
                                        query = args.get("query", "")
                                        result = msg.content or ""
                                        break
                                if query:
                                    break
                    if query:
                        break

            if query:
                candidates.append({"query": query, "result": result})

        if not candidates:
            yield AgentEvent(
                type=EventType.AGENT_ERROR,
                data={"error": "Draft phase did not produce any SQL query"},
            )
            return

        # ── Review Phase — pick best candidate, fix if needed ──
        review_tools: dict[str, Tool] = {}
        for name in ("execute_sql", "submit_answer", "sample_values",
                      "describe_column"):
            if name in self.tools:
                review_tools[name] = self.tools[name]

        # Show all rules with tags so reviewer knows which are FILTER vs TRANSFORM
        tagged_rules_block = _format_rules(all_rules, show_tags=True)

        # Build candidate blocks with sanity checks
        candidate_blocks: list[str] = []
        for i, c in enumerate(candidates):
            label = chr(ord("A") + i)
            sanity = _run_sanity_checks(c["query"], c["result"])
            block = (
                f"### Candidate {label}\n"
                f"```sql\n{c['query']}\n```\n"
                f"Results: {c['result'][:1500]}\n"
            )
            if sanity:
                block += f"\n{sanity}\n"
            candidate_blocks.append(block)

        # Build structured comparison if we have two candidates
        comparison_block = ""
        if len(candidates) == 2:
            comparison_block = _compare_candidates(
                candidates[0], candidates[1],
            )

        review_content = (
            f"## Question\n{prompt}\n\n"
            f"## Candidates\n"
            + "\n".join(candidate_blocks)
            + f"\n{comparison_block}"
            f"{tagged_rules_block}"
            f"{gathered_context}\n"
        )

        review_conversation = Conversation()
        review_conversation.messages.append(
            Message(role="system", content=self._REVIEW_SYSTEM_PROMPT)
        )
        review_conversation.messages.append(
            Message(role="user", content=review_content)
        )

        for item in self._run_phase(
            review_conversation, review_tools, review_budget, "sql"
        ):
            if isinstance(item, str):
                pass
            else:
                yield item

    def reset_conversation(self) -> None:
        """Reset the conversation to the initial state."""
        self.conversation = Conversation()


def _parse_tool_calls_from_api(tool_calls_data: list[dict[str, Any]]) -> list[ToolCall]:
    """Parse tool calls from OpenAI-compatible API response format."""
    tool_calls: list[ToolCall] = []

    for tc in tool_calls_data:
        tc_id = tc.get("id", "")
        function = tc.get("function", {})
        name = function.get("name", "")
        arguments_str = function.get("arguments", "{}")

        try:
            arguments = json.loads(arguments_str)
            error = None
        except json.JSONDecodeError as e:
            # Don't print to stdout, return error in ToolCall
            arguments = {}
            error = f"Invalid JSON arguments: {e}"

        tool_calls.append(
            ToolCall(
                id=tc_id,
                name=name,
                arguments=arguments,
                error=error,
            )
        )

    return tool_calls
