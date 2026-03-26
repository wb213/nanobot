"""AWS Bedrock provider — Anthropic SDK with Bedrock backend."""

from __future__ import annotations

import os
from typing import Any

from loguru import logger

from nanobot.providers.anthropic_provider import AnthropicProvider
from nanobot.providers.base import LLMResponse


class AWSBedrockProvider(AnthropicProvider):
    """LLM provider using Anthropic SDK with AWS Bedrock backend.

    Inherits all Claude functionality from AnthropicProvider but uses
    AnthropicBedrock client to route requests through AWS Bedrock.

    Uses boto3's standard credential chain:
    1. Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_BEARER_TOKEN_BEDROCK)
    2. AWS config files (~/.aws/credentials, ~/.aws/config)
    3. IAM role (EC2, ECS, Lambda, EKS)
    4. AWS SSO

    Region resolution order:
    1. region_name parameter
    2. AWS_REGION environment variable
    3. AWS_DEFAULT_REGION environment variable
    4. boto3 default
    """

    def __init__(
        self,
        region_name: str | None = None,
        default_model: str = "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
        extra_headers: dict[str, str] | None = None,
    ):
        """Initialize Bedrock provider.

        Args:
            region_name: AWS region (optional, falls back to boto3 default)
            default_model: Bedrock model ID (format: global.anthropic.claude-*, suggest use global inference profile)
            extra_headers: Additional HTTP headers
        """
        # Don't call super().__init__() - we need different client setup
        from nanobot.providers.base import LLMProvider

        LLMProvider.__init__(self, api_key=None, api_base=None)
        self.default_model = default_model
        self.extra_headers = extra_headers or {}

        try:
            from anthropic import AsyncAnthropicBedrock
        except ImportError as e:
            raise ImportError(
                "AWS Bedrock support requires: pip install nanobot-ai[bedrock]\n"
                "Or: pip install anthropic boto3"
            ) from e

        # Region resolution: parameter > AWS_REGION > AWS_DEFAULT_REGION > boto3 default
        region = region_name or os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")

        client_kw: dict[str, Any] = {}
        if region:
            client_kw["aws_region"] = region
        if extra_headers:
            client_kw["default_headers"] = extra_headers

        # boto3 handles all credential resolution automatically
        self._client = AsyncAnthropicBedrock(**client_kw)

    @staticmethod
    def _strip_prefix(model: str) -> str:
        if model.startswith("bedrock/"):
            return model[len("bedrock/"):]
        return model

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
        """Chat completion via AWS Bedrock.

        Bedrock supports prompt caching for Claude models.
        Cache control is enabled by default (inherited from AnthropicProvider).
        """
        kwargs = self._build_kwargs(
            messages, tools, model, max_tokens, temperature,
            reasoning_effort, tool_choice,
            supports_caching=True,
        )
        try:
            response = await self._client.messages.create(**kwargs)
            parsed = self._parse_response(response)
            if parsed.usage:
                actual_prompt = parsed.usage.get("prompt_tokens", 0)
                logger.debug(f"Bedrock actual usage: {actual_prompt} prompt tokens, "
                           f"{parsed.usage.get('completion_tokens', 0)} completion tokens")
            return parsed
        except Exception as e:
            return LLMResponse(content=f"Error calling LLM: {e}", finish_reason="error")

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
        """Streaming chat completion via AWS Bedrock.

        Bedrock supports prompt caching for Claude models.
        Cache control is enabled by default (inherited from AnthropicProvider).
        """
        kwargs = self._build_kwargs(
            messages, tools, model, max_tokens, temperature,
            reasoning_effort, tool_choice,
            supports_caching=True,
        )
        try:
            async with self._client.messages.stream(**kwargs) as stream:
                if on_content_delta:
                    async for text in stream.text_stream:
                        await on_content_delta(text)
                response = await stream.get_final_message()
            parsed = self._parse_response(response)
            if parsed.usage:
                actual_prompt = parsed.usage.get("prompt_tokens", 0)
                logger.debug(f"Bedrock actual usage: {actual_prompt} prompt tokens, "
                           f"{parsed.usage.get('completion_tokens', 0)} completion tokens")
            return parsed
        except Exception as e:
            return LLMResponse(content=f"Error calling LLM: {e}", finish_reason="error")

    async def estimate_prompt_tokens(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
    ) -> tuple[int, str]:
        """Use Anthropic's count_tokens API for accurate token estimation."""
        try:
            # Reuse _build_kwargs to get properly formatted params
            kwargs = self._build_kwargs(
                messages, tools, model,
                max_tokens=1024,  # dummy value, not used by count_tokens
                temperature=0.0,  # dummy value, not used by count_tokens
                reasoning_effort=None,
                tool_choice=None,
                supports_caching=False,  # don't need cache control for counting
            )

            # Extract only what count_tokens needs
            count_kwargs = {
                "model": kwargs["model"],
                "messages": kwargs["messages"],
            }
            if "system" in kwargs:
                count_kwargs["system"] = kwargs["system"]
            if "tools" in kwargs:
                count_kwargs["tools"] = kwargs["tools"]

            result = await self._client.messages.count_tokens(**count_kwargs)
            estimated = result.input_tokens
            logger.debug(f"Bedrock token estimation: {estimated} tokens (via count_tokens API)")
            return estimated, "bedrock_count_tokens"
        except Exception as e:
            logger.debug(f"Bedrock token counting failed: {e}")
            return 0, "failed"

    def get_default_model(self) -> str:
        return self.default_model
