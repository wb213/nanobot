"""AWS Bedrock provider — Native boto3 implementation with Converse API.

Based on Strands Agents SDK bedrock.py implementation.
Simplified to focus on core Converse API functionality.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, cast

import boto3
from botocore.config import Config as BotocoreConfig
from botocore.exceptions import ClientError
from loguru import logger

from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest

DEFAULT_BEDROCK_MODEL_ID = "us.anthropic.claude-sonnet-4-20250514-v1:0"
DEFAULT_BEDROCK_REGION = "us-west-2"
DEFAULT_READ_TIMEOUT = 120

BEDROCK_CONTEXT_WINDOW_OVERFLOW_MESSAGES = [
    "Input is too long for requested model",
    "input length and `max_tokens` exceed context limit",
    "too many total text bytes",
    "prompt is too long",
]

# Models that should include tool result status field
_MODELS_INCLUDE_STATUS = ["anthropic.claude"]


class AWSBedrockProvider(LLMProvider):
    """LLM provider using AWS Bedrock with native boto3 Converse API.

    Uses boto3's standard credential chain:
    1. Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN)
    2. AWS config files (~/.aws/credentials, ~/.aws/config)
    3. IAM role (EC2, ECS, Lambda, EKS)
    4. AWS SSO

    Region resolution order:
    1. region_name parameter
    2. AWS_REGION environment variable
    3. AWS_DEFAULT_REGION environment variable
    4. boto3 default (us-west-2)
    """

    def __init__(
        self,
        region_name: str | None = None,
        default_model: str = DEFAULT_BEDROCK_MODEL_ID,
        extra_headers: dict[str, str] | None = None,
    ):
        """Initialize Bedrock provider.

        Args:
            region_name: AWS region (optional, falls back to boto3 default)
            default_model: Bedrock model ID or inference profile
            extra_headers: Additional HTTP headers (not used by boto3)
        """
        super().__init__(api_key=None, api_base=None)
        self.default_model = default_model
        self.extra_headers = extra_headers or {}

        try:
            import boto3
        except ImportError as e:
            raise ImportError(
                "AWS Bedrock support requires: pip install nanobot-ai[bedrock]\n"
                "Or: pip install boto3"
            ) from e

        # Region resolution: parameter > AWS_REGION > AWS_DEFAULT_REGION > boto3 default
        session = boto3.Session()
        resolved_region = (
            region_name
            or os.getenv("AWS_REGION")
            or os.getenv("AWS_DEFAULT_REGION")
            or session.region_name
            or DEFAULT_BEDROCK_REGION
        )

        # Add strands-agents style user agent for tracking
        client_config = BotocoreConfig(
            user_agent_extra="nanobot-ai",
            read_timeout=DEFAULT_READ_TIMEOUT,
        )

        self.client = session.client(
            service_name="bedrock-runtime",
            config=client_config,
            region_name=resolved_region,
        )

        logger.debug(f"Bedrock client created for region: {self.client.meta.region_name}")

    @staticmethod
    def _strip_prefix(model: str) -> str:
        """Remove bedrock/ prefix from model name if present."""
        if model.startswith("bedrock/"):
            return model[len("bedrock/"):]
        return model

    @property
    def _cache_strategy(self) -> str | None:
        """The cache strategy for this model based on its model ID."""
        model_id = self.default_model.lower()
        if "claude" in model_id or "anthropic" in model_id:
            return "anthropic"
        return None

    def get_default_model(self) -> str:
        return self.default_model

    def _should_include_tool_result_status(self) -> bool:
        """Determine whether to include tool result status based on model."""
        model_id = self.default_model.lower()
        return any(model in model_id for model in _MODELS_INCLUDE_STATUS)

    def _inject_cache_point(self, messages: list[dict[str, Any]]) -> None:
        """Inject a cache point at the end of the last user message.

        Args:
            messages: List of messages to inject cache point into (modified in place).
        """
        if not messages:
            return

        # Remove any existing cache points first
        for msg in messages:
            content = msg.get("content", [])
            for block_idx in reversed(range(len(content))):
                if isinstance(content[block_idx], dict) and "cachePoint" in content[block_idx]:
                    del content[block_idx]

        # Find last user message
        last_user_idx = None
        for msg_idx, msg in enumerate(messages):
            if msg.get("role") == "user":
                last_user_idx = msg_idx

        # Add cache point to last user message
        if last_user_idx is not None and messages[last_user_idx].get("content"):
            messages[last_user_idx]["content"].append({"cachePoint": {"type": "default"}})
            logger.debug(f"Added cache point to message index {last_user_idx}")

    def _format_request_message_content(self, content: dict[str, Any]) -> dict[str, Any] | None:
        """Format a Bedrock content block.

        Bedrock strictly validates content blocks and throws exceptions for unknown fields.
        This extracts only the fields that Bedrock supports for each content type.

        Args:
            content: Content block to format.

        Returns:
            Bedrock formatted content block, or None if unsupported/filtered.
        """
        # Cache point
        if "cachePoint" in content:
            return {"cachePoint": {"type": content["cachePoint"]["type"]}}

        # Text content
        if "text" in content:
            return {"text": content["text"]}

        # Image content
        if "image" in content:
            image = content["image"]
            source = image["source"]
            if "bytes" in source:
                return {
                    "image": {
                        "format": image["format"],
                        "source": {"bytes": source["bytes"]}
                    }
                }
            # Skip S3 location sources (not implementing)
            logger.warning("S3 image sources not supported, skipping")
            return None

        # Document content
        if "document" in content:
            document = content["document"]
            result: dict[str, Any] = {}

            if "name" in document:
                result["name"] = document["name"]
            if "format" in document:
                result["format"] = document["format"]

            # Handle source
            if "source" in document:
                source = document["source"]
                if "bytes" in source:
                    result["source"] = {"bytes": source["bytes"]}
                else:
                    # Skip S3 location sources
                    logger.warning("S3 document sources not supported, skipping")
                    return None

            return {"document": result}

        # Reasoning content (thinking blocks)
        if "reasoningContent" in content:
            reasoning = content["reasoningContent"]
            result = {}

            if "reasoningText" in reasoning:
                reasoning_text = reasoning["reasoningText"]
                result["reasoningText"] = {}
                if "text" in reasoning_text:
                    result["reasoningText"]["text"] = reasoning_text["text"]
                # CRITICAL: Include signature if present
                if reasoning_text.get("signature"):
                    result["reasoningText"]["signature"] = reasoning_text["signature"]

            return {"reasoningContent": result}

        # Tool use
        if "toolUse" in content:
            tool_use = content["toolUse"]
            return {
                "toolUse": {
                    "input": tool_use["input"],
                    "name": tool_use["name"],
                    "toolUseId": tool_use["toolUseId"],
                }
            }

        # Tool result
        if "toolResult" in content:
            tool_result = content["toolResult"]
            formatted_content: list[dict[str, Any]] = []

            for tool_result_content in tool_result["content"]:
                if "json" in tool_result_content:
                    # Special case: json field valid in ToolResultContent
                    formatted_content.append({"json": tool_result_content["json"]})
                else:
                    # Recursively format other content blocks
                    formatted = self._format_request_message_content(tool_result_content)
                    if formatted is not None:
                        formatted_content.append(formatted)

            result = {
                "content": formatted_content,
                "toolUseId": tool_result["toolUseId"],
            }

            # Include status field for Claude models
            if "status" in tool_result and self._should_include_tool_result_status():
                result["status"] = tool_result["status"]

            return {"toolResult": result}

        # Unsupported content type
        content_type = next(iter(content)) if content else "unknown"
        logger.warning(f"Unsupported content type: {content_type}, skipping")
        return None

    def _format_bedrock_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Format messages for Bedrock API compatibility.

        Filters content blocks to only include Bedrock-supported fields and
        injects cache points when automatic caching is enabled.

        Args:
            messages: List of messages to format

        Returns:
            Messages formatted for Bedrock API
        """
        cleaned_messages: list[dict[str, Any]] = []

        for message in messages:
            content = message.get("content")

            # Handle different content types from _sanitize_empty_content
            if isinstance(content, str):
                # String content (e.g., "(empty)" or regular text)
                cleaned_content = [{"text": content}]
            elif isinstance(content, list):
                # List of content blocks
                cleaned_content = []
                for content_block in content:
                    if isinstance(content_block, dict):
                        formatted = self._format_request_message_content(content_block)
                        if formatted is not None:
                            cleaned_content.append(formatted)
                    elif isinstance(content_block, str):
                        # String item in list
                        cleaned_content.append({"text": content_block})
            elif content is None:
                # None content (assistant with tool_calls only)
                cleaned_content = []
            else:
                # Fallback: convert to string
                cleaned_content = [{"text": str(content)}]

            # Add message if it has content
            if cleaned_content:
                cleaned_messages.append({
                    "content": cleaned_content,
                    "role": message["role"]
                })

        # Auto-inject cache point for anthropic models
        if self._cache_strategy == "anthropic":
            self._inject_cache_point(cleaned_messages)

        return cleaned_messages

    def _format_bedrock_tools(self, tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
        """Format OpenAI-style tools to Bedrock format.

        Args:
            tools: OpenAI format tool definitions

        Returns:
            Bedrock format tool specs
        """
        if not tools:
            return None

        bedrock_tools = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool["function"]
                schema = func.get("parameters", {"type": "object", "properties": {}})

                # Bedrock requires inputSchema to be wrapped in {"json": <schema>}
                bedrock_tools.append({
                    "toolSpec": {
                        "name": func["name"],
                        "description": func.get("description", ""),
                        "inputSchema": {"json": schema}
                    }
                })

        return bedrock_tools if bedrock_tools else None

    def _convert_tool_choice(self, tool_choice: str | dict[str, Any] | None) -> dict[str, Any]:
        """Convert OpenAI tool_choice to Bedrock format.

        Args:
            tool_choice: OpenAI format ("auto", "required", or {"type": "function", "function": {...}})

        Returns:
            Bedrock format tool choice
        """
        if tool_choice is None or tool_choice == "auto":
            return {"auto": {}}
        elif tool_choice == "required" or tool_choice == "any":
            return {"any": {}}
        elif isinstance(tool_choice, dict):
            if tool_choice.get("type") == "function":
                func_name = tool_choice.get("function", {}).get("name")
                if func_name:
                    return {"tool": {"name": func_name}}
        return {"auto": {}}

    def _format_request(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Format a Bedrock converse request.

        Args:
            messages: List of message objects
            tools: List of tool definitions
            model: Model ID
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            reasoning_effort: Extended thinking effort level
            tool_choice: Tool selection strategy

        Returns:
            A Bedrock converse request dict
        """
        # Extract system messages
        system_blocks = []
        regular_messages = []

        for msg in messages:
            if msg.get("role") == "system":
                # System messages go to system parameter
                content = msg.get("content")
                if isinstance(content, str):
                    system_blocks.append({"text": content})
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and "text" in block:
                            system_blocks.append({"text": block["text"]})
            else:
                regular_messages.append(msg)

        # Format messages
        formatted_messages = self._format_bedrock_messages(regular_messages)

        # Format tools
        bedrock_tools = self._format_bedrock_tools(tools)

        # Build request
        request: dict[str, Any] = {
            "modelId": self._strip_prefix(model or self.default_model),
            "messages": formatted_messages,
            "inferenceConfig": {
                "maxTokens": max_tokens,
                "temperature": temperature,
            }
        }

        # Add system if present
        if system_blocks:
            request["system"] = system_blocks

        # Add tools if present
        if bedrock_tools:
            tool_config: dict[str, Any] = {"tools": bedrock_tools}
            if tool_choice is not None:
                tool_config["toolChoice"] = self._convert_tool_choice(tool_choice)
            request["toolConfig"] = tool_config

        # Add thinking config if reasoning_effort specified
        if reasoning_effort:
            budget_map = {"low": 1024, "medium": 4096, "high": 8192}
            budget = budget_map.get(reasoning_effort.lower(), 4096)
            request["additionalModelRequestFields"] = {
                "thinking": {
                    "type": "enabled",
                    "budget_tokens": budget
                }
            }

        return request

    def _parse_non_streaming_response(self, response: dict[str, Any]) -> LLMResponse:
        """Parse non-streaming Bedrock response into LLMResponse.

        Args:
            response: Bedrock converse API response

        Returns:
            Parsed LLMResponse
        """
        content_parts: list[str] = []
        tool_calls: list[ToolCallRequest] = []
        thinking_blocks: list[dict[str, Any]] = []
        reasoning_text: str | None = None

        # Parse content blocks
        for block in response["output"]["message"].get("content", []):
            if "text" in block:
                content_parts.append(block["text"])
            elif "toolUse" in block:
                tool_use = block["toolUse"]
                tool_calls.append(ToolCallRequest(
                    id=tool_use["toolUseId"],
                    name=tool_use["name"],
                    arguments=tool_use["input"] if isinstance(tool_use["input"], dict) else {}
                ))
            elif "reasoningContent" in block:
                reasoning = block["reasoningContent"]
                if "reasoningText" in reasoning:
                    reasoning_text_block = reasoning["reasoningText"]
                    text = reasoning_text_block.get("text", "")
                    signature = reasoning_text_block.get("signature", "")

                    # Store for reasoning_content field
                    if text and reasoning_text is None:
                        reasoning_text = text

                    # Store in thinking_blocks for round-trip
                    thinking_blocks.append({
                        "type": "thinking",
                        "thinking": text,
                        "signature": signature
                    })

        # Map finish reason
        stop_reason_map = {
            "end_turn": "stop",
            "tool_use": "tool_calls",
            "max_tokens": "length"
        }
        finish_reason = stop_reason_map.get(
            response.get("stopReason", "end_turn"),
            response.get("stopReason", "stop")
        )

        # Parse usage
        usage = {}
        if "usage" in response:
            usage = {
                "prompt_tokens": response["usage"].get("inputTokens", 0),
                "completion_tokens": response["usage"].get("outputTokens", 0),
                "total_tokens": response["usage"].get("totalTokens", 0)
            }

        return LLMResponse(
            content="".join(content_parts) if content_parts else None,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage,
            reasoning_content=reasoning_text,
            thinking_blocks=thinking_blocks if thinking_blocks else None
        )

    def _call_bedrock_sync(
        self,
        request: dict[str, Any],
        streaming: bool = False,
    ) -> dict[str, Any] | Any:
        """Synchronous Bedrock API call (runs in thread).

        Args:
            request: Formatted Bedrock request
            streaming: Whether to use streaming API

        Returns:
            Bedrock response (dict for non-streaming, stream for streaming)

        Raises:
            ClientError: On Bedrock API errors
        """
        try:
            if streaming:
                response = self.client.converse_stream(**request)
                return response
            else:
                response = self.client.converse(**request)
                return response
        except ClientError as e:
            error_message = str(e)

            # Check for context window overflow
            if any(msg in error_message for msg in BEDROCK_CONTEXT_WINDOW_OVERFLOW_MESSAGES):
                logger.warning("Bedrock context window overflow")
                raise

            # Add debug info
            logger.error(
                f"Bedrock API error: {e.response['Error']['Code']} - {error_message}\n"
                f"Region: {self.client.meta.region_name}\n"
                f"Model: {request.get('modelId')}"
            )
            raise

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Chat completion via AWS Bedrock Converse API.

        Args:
            messages: List of message dicts
            tools: Optional tool definitions
            model: Model ID
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            reasoning_effort: Extended thinking effort ("low", "medium", "high")
            tool_choice: Tool selection strategy

        Returns:
            LLMResponse with content and/or tool calls
        """
        messages = self._sanitize_empty_content(messages)

        request = self._format_request(
            messages=messages,
            tools=tools,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            tool_choice=tool_choice,
        )

        logger.debug(f"Bedrock request: {json.dumps(request, default=str)[:500]}")

        # Call Bedrock in thread
        try:
            response = await asyncio.to_thread(
                self._call_bedrock_sync,
                request,
                streaming=False
            )

            logger.debug(f"Bedrock response: {json.dumps(response, default=str)[:500]}")
            return self._parse_non_streaming_response(response)

        except ClientError as e:
            error_msg = f"Bedrock error: {e.response['Error']['Code']} - {str(e)}"
            logger.error(error_msg)
            return LLMResponse(content=error_msg, finish_reason="error")
        except Exception as e:
            error_msg = f"Unexpected error: {type(e).__name__}: {str(e)}"
            logger.exception(error_msg)
            return LLMResponse(content=error_msg, finish_reason="error")

    async def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        on_content_delta: Any = None,
    ) -> LLMResponse:
        """Streaming chat completion via AWS Bedrock Converse API.

        Uses asyncio.to_thread() to wrap synchronous boto3 streaming in async context.

        Args:
            messages: List of message dicts
            tools: Optional tool definitions
            model: Model ID
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            reasoning_effort: Extended thinking effort ("low", "medium", "high")
            tool_choice: Tool selection strategy
            on_content_delta: Callback for each text chunk

        Returns:
            Final LLMResponse
        """
        messages = self._sanitize_empty_content(messages)

        request = self._format_request(
            messages=messages,
            tools=tools,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            tool_choice=tool_choice,
        )

        logger.debug(f"Bedrock streaming request: {json.dumps(request, default=str)[:500]}")

        # Accumulate response parts
        content_parts: list[str] = []
        tool_calls: list[ToolCallRequest] = []
        thinking_blocks: list[dict[str, Any]] = []
        reasoning_text_parts: list[str] = []
        current_signature: str = ""
        stop_reason: str = "stop"
        usage: dict[str, int] = {}

        # Current tool use being built
        current_tool: dict[str, Any] | None = None

        def process_stream_sync():
            """Process stream in thread (sync context)."""
            nonlocal stop_reason, usage, current_tool, current_signature

            try:
                response = self._call_bedrock_sync(request, streaming=True)

                for chunk in response["stream"]:
                    # Message start
                    if "messageStart" in chunk:
                        pass  # role info, not needed

                    # Content block start
                    elif "contentBlockStart" in chunk:
                        start = chunk["contentBlockStart"].get("start", {})
                        if "toolUse" in start:
                            # Start accumulating tool use
                            tool_start = start["toolUse"]
                            current_tool = {
                                "id": tool_start["toolUseId"],
                                "name": tool_start["name"],
                                "input_chunks": []
                            }

                    # Content block delta
                    elif "contentBlockDelta" in chunk:
                        delta = chunk["contentBlockDelta"].get("delta", {})

                        if "text" in delta:
                            # Text content
                            text = delta["text"]
                            content_parts.append(text)
                            if on_content_delta:
                                # Queue for async callback
                                loop.call_soon_threadsafe(
                                    lambda: asyncio.create_task(on_content_delta(text))
                                )

                        elif "toolUse" in delta:
                            # Tool use input chunk
                            if current_tool:
                                current_tool["input_chunks"].append(delta["toolUse"]["input"])

                        elif "reasoningContent" in delta:
                            # Reasoning/thinking content
                            reasoning = delta["reasoningContent"]
                            if "text" in reasoning:
                                reasoning_text_parts.append(reasoning["text"])
                            if "signature" in reasoning:
                                # Signature comes at the end
                                current_signature += reasoning["signature"]

                    # Content block stop
                    elif "contentBlockStop" in chunk:
                        # Finalize current tool use
                        if current_tool:
                            input_str = "".join(current_tool["input_chunks"])
                            try:
                                input_dict = json.loads(input_str)
                            except json.JSONDecodeError:
                                input_dict = {}

                            tool_calls.append(ToolCallRequest(
                                id=current_tool["id"],
                                name=current_tool["name"],
                                arguments=input_dict
                            ))
                            current_tool = None

                        # Finalize thinking block
                        if reasoning_text_parts:
                            thinking_blocks.append({
                                "type": "thinking",
                                "thinking": "".join(reasoning_text_parts),
                                "signature": current_signature
                            })
                            reasoning_text_parts.clear()
                            current_signature = ""

                    # Message stop
                    elif "messageStop" in chunk:
                        stop_info = chunk["messageStop"]
                        stop_reason_raw = stop_info.get("stopReason", "end_turn")
                        stop_reason_map = {
                            "end_turn": "stop",
                            "tool_use": "tool_calls",
                            "max_tokens": "length"
                        }
                        stop_reason = stop_reason_map.get(stop_reason_raw, stop_reason_raw)

                    # Metadata (usage info)
                    elif "metadata" in chunk:
                        metadata = chunk["metadata"]
                        if "usage" in metadata:
                            usage = {
                                "prompt_tokens": metadata["usage"].get("inputTokens", 0),
                                "completion_tokens": metadata["usage"].get("outputTokens", 0),
                                "total_tokens": metadata["usage"].get("totalTokens", 0)
                            }

            except Exception as e:
                logger.exception("Error processing Bedrock stream")
                raise

        # Run stream processing in thread
        try:
            loop = asyncio.get_event_loop()
            await asyncio.to_thread(process_stream_sync)

            # Build final response
            reasoning_content = "".join(reasoning_text_parts) if reasoning_text_parts else None
            if not reasoning_content and thinking_blocks:
                # Extract from first thinking block if not accumulated during streaming
                reasoning_content = thinking_blocks[0].get("thinking")

            return LLMResponse(
                content="".join(content_parts) if content_parts else None,
                tool_calls=tool_calls,
                finish_reason=stop_reason,
                usage=usage,
                reasoning_content=reasoning_content,
                thinking_blocks=thinking_blocks if thinking_blocks else None
            )

        except ClientError as e:
            error_msg = f"Bedrock streaming error: {e.response['Error']['Code']} - {str(e)}"
            logger.error(error_msg)
            return LLMResponse(content=error_msg, finish_reason="error")
        except Exception as e:
            error_msg = f"Unexpected streaming error: {type(e).__name__}: {str(e)}"
            logger.exception(error_msg)
            return LLMResponse(content=error_msg, finish_reason="error")
