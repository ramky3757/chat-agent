"""
Hermes agent — agentic loop that drives Claude with MySQL tools.

Prompt caching is applied to both the system prompt and the tools
definition so repeated turns in the same session reuse the KV cache
and cut latency/cost.
"""

import json
import time

import anthropic

from db_tool import BaseDBTool


_SYSTEM_PROMPT = """\
You are a helpful data analyst assistant with direct, read-only access to a database.

Your job:
- Understand the user's natural-language question about data
- If domain knowledge is provided in the system context, use it as your PRIMARY reference for
  table names, column names, relationships, joins, and business context — trust it over assumptions
- Only call `list_tables` or `describe_table` for tables NOT already covered in the domain knowledge
- Write a precise SQL query (SELECT / SHOW / DESCRIBE / EXPLAIN only) and execute it with `execute_sql_query`
- Present results clearly (tables, bullet lists, or prose depending on context)
- Briefly show the SQL you ran so users can learn from it
- Always add a LIMIT clause whenever result size is unpredictable

PERFORMANCE RULE:
- Before executing queries that scan large tables (no WHERE clause, no date/status filter), warn the user
  and suggest adding filters (date ranges, status, IDs) for faster results. Ask them to confirm or narrow first.

HARD RULES — never violate these:
• Only READ operations are allowed. Do NOT attempt INSERT, UPDATE, DELETE, DROP,
  TRUNCATE, ALTER, CREATE, REPLACE, GRANT, or any other write/DDL statement.
• The database layer will reject write operations, but you must not even attempt them.
• If the user asks for a write operation, politely explain it is not permitted.\
"""

_TOOLS: list[dict] = [
    {
        "name": "list_tables",
        "description": (
            "List all table names in the database. "
            "Call this first to discover which tables exist before building a query."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "describe_table",
        "description": (
            "Get column definitions (Field, Type, Null, Key, Default, Extra) for a single table. "
            "Call this once you know which table you need — do NOT load the full schema at once."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "table_name": {
                    "type": "string",
                    "description": "The exact table name to describe.",
                }
            },
            "required": ["table_name"],
        },
    },
    {
        "name": "execute_sql_query",
        "description": (
            "Execute a READ-ONLY SQL statement on the MySQL database. "
            "Accepted: SELECT, SHOW, DESCRIBE, EXPLAIN, CTEs (WITH …). "
            "Returns: columns, rows, row_count, truncation note if capped. "
            "Write operations are rejected."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The read-only SQL statement to run.",
                }
            },
            "required": ["query"],
        },
        "cache_control": {"type": "ephemeral"},
    },
]

_MAX_RETRIES = 3
_RETRY_DELAY = 15  # seconds to wait on a 429


class HermesAgent:
    """Agentic Claude loop with DB tool access and multi-turn session memory."""

    def __init__(self, config: dict, db_tool: BaseDBTool, skills_text: str = "") -> None:
        self.client = anthropic.Anthropic()
        self.db_tool = db_tool
        self.model: str = config["claude"]["model"]
        self.max_tokens: int = int(config["claude"].get("max_tokens", 1024))
        self.max_history: int = int(config["session"].get("max_history_messages", 10))
        self.conversation: list[dict] = []
        self._table_cache: dict | None = None
        self._skills_text: str = skills_text.strip()

    # ------------------------------------------------------------------

    def _trim(self) -> None:
        while len(self.conversation) > self.max_history:
            self.conversation.pop(0)
            if self.conversation and self.conversation[0]["role"] == "assistant":
                self.conversation.pop(0)

    def _run_tool(self, name: str, tool_input: dict) -> str:
        if name == "execute_sql_query":
            result = self.db_tool.execute_query(tool_input.get("query", ""))
        elif name == "list_tables":
            if self._table_cache is None:
                self._table_cache = self.db_tool.list_tables()
            result = self._table_cache
        elif name == "describe_table":
            result = self.db_tool.describe_table(tool_input.get("table_name", ""))
        else:
            result = {"error": f"Unknown tool '{name}'."}
        return json.dumps(result, default=str)

    def _make_request_kwargs(self, messages: list) -> dict:
        system: list[dict] = [
            {
                "type": "text",
                "text": _SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ]
        if self._skills_text:
            system.append(
                {
                    "type": "text",
                    "text": f"## Domain Knowledge & Schema Context\n\n{self._skills_text}",
                    "cache_control": {"type": "ephemeral"},
                }
            )
        return dict(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            tools=_TOOLS,
            messages=messages,
        )

    # ------------------------------------------------------------------
    # Blocking chat (used by CLI fallback and /chat API endpoint)
    # ------------------------------------------------------------------

    def chat(self, user_message: str) -> str:
        self.conversation.append({"role": "user", "content": user_message})
        self._trim()
        messages = list(self.conversation)

        while True:
            response = self._call_with_retry(messages)

            if response.stop_reason == "tool_use":
                messages.append({"role": "assistant", "content": response.content})
                tool_results = [
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": self._run_tool(block.name, block.input),
                    }
                    for block in response.content
                    if block.type == "tool_use"
                ]
                messages.append({"role": "user", "content": tool_results})
                continue

            final_text = "".join(
                block.text for block in response.content if hasattr(block, "text")
            )
            self.conversation.append({"role": "assistant", "content": final_text})
            return final_text

    def _call_with_retry(self, messages: list) -> anthropic.types.Message:
        for attempt in range(_MAX_RETRIES):
            try:
                return self.client.messages.create(**self._make_request_kwargs(messages))
            except anthropic.RateLimitError:
                if attempt == _MAX_RETRIES - 1:
                    raise
                time.sleep(_RETRY_DELAY * (attempt + 1))

    # ------------------------------------------------------------------
    # Streaming chat — yields text chunks as Claude generates them
    # ------------------------------------------------------------------

    def chat_stream(self, user_message: str):
        """
        Generator: yields str chunks as Claude writes the response.
        Tool calls execute silently in the background; only the final
        text response (and any inline "thinking" text) is yielded.
        """
        self.conversation.append({"role": "user", "content": user_message})
        self._trim()
        messages = list(self.conversation)
        accumulated: list[str] = []

        while True:
            turn_chunks: list[str] = []
            final = None

            for attempt in range(_MAX_RETRIES):
                turn_chunks.clear()
                try:
                    with self.client.messages.stream(
                        **self._make_request_kwargs(messages)
                    ) as stream:
                        for chunk in stream.text_stream:
                            turn_chunks.append(chunk)
                            yield chunk
                        final = stream.get_final_message()
                    break
                except anthropic.RateLimitError:
                    if attempt == _MAX_RETRIES - 1:
                        raise
                    time.sleep(_RETRY_DELAY * (attempt + 1))

            if final.stop_reason == "tool_use":
                messages.append({"role": "assistant", "content": final.content})
                tool_results = [
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": self._run_tool(block.name, block.input),
                    }
                    for block in final.content
                    if block.type == "tool_use"
                ]
                messages.append({"role": "user", "content": tool_results})
                accumulated.extend(turn_chunks)
                continue

            accumulated.extend(turn_chunks)
            self.conversation.append(
                {"role": "assistant", "content": "".join(accumulated)}
            )
            return

    # ------------------------------------------------------------------

    def reset(self) -> None:
        self.conversation.clear()
        self._table_cache = None

    @property
    def history_length(self) -> int:
        return len(self.conversation)
