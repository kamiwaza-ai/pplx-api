import os
from typing import List, Dict, Optional, Union, Any, Callable, Awaitable
import requests
from pydantic import BaseModel, Field, field_validator
from pydantic_core import PydanticCustomError
import json
import asyncio
from functools import partial
import aiohttp
import backoff
import logging

logger = logging.getLogger(__name__)

class Message(BaseModel):
    role: str
    content: str

class PerplexityRequest(BaseModel):
    model: str = Field(default="sonar")
    messages: List[Message]
    max_tokens: Optional[int] = None
    temperature: float = Field(default=0.2, ge=0, lt=2)
    top_p: float = Field(default=0.9, ge=0, le=1)
    return_citations: bool = Field(default=True)
    search_domain_filter: Optional[List[str]] = Field(default=None, max_length=3)
    return_images: bool = Field(default=False)
    return_related_questions: bool = Field(default=False)
    search_recency_filter: Optional[str] = Field(default=None, description="Filters search results based on time (e.g., 'week', 'day')")
    top_k: int = Field(default=0, ge=0, le=2048)
    stream: bool = Field(default=True)
    presence_penalty: float = Field(default=0, ge=-2, le=2)
    frequency_penalty: float = Field(default=1, ge=0)
    response_format: Optional[Dict[str, Any]] = Field(default=None)
    web_search_options: Optional[Dict[str, str]] = Field(default=None)

    @field_validator('search_recency_filter')
    def validate_search_recency_filter(cls, v):
        if v is not None and v not in ['month', 'week', 'day', 'hour']:
            raise PydanticCustomError(
                'invalid_search_recency_filter',
                'search_recency_filter must be one of: month, week, day, hour'
            )
        return v

    @field_validator('web_search_options')
    def validate_web_search_options(cls, v):
        if v is not None and 'search_context_size' in v:
            if v['search_context_size'] not in ['low', 'medium', 'high']:
                raise PydanticCustomError(
                    'invalid_search_context_size',
                    'search_context_size must be one of: low, medium, high'
                )
        return v

    model_config = {
        'extra': 'forbid'
    }

class PerplexityError(Exception):
    """Base exception for Perplexity API errors."""
    pass

class PerplexityClient:
    """A client for interacting with the Perplexity API."""

    BASE_URL = "https://api.perplexity.ai/chat/completions"
    MAX_RETRIES = 2
    INITIAL_WAIT = 1  # Initial wait in seconds
    MAX_WAIT = 4      # Max wait in seconds

    def __init__(self, api_key: Optional[str] = None, debug: bool = False):
        self.api_key = api_key or os.environ.get("PERPLEXITY_API_KEY")
        self.debug = debug
        if not self.api_key:
            raise ValueError("API key must be provided either as a parameter or through the PERPLEXITY_API_KEY environment variable.")

    def _get_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    async def _stream_response(
        self,
        response: Union[requests.Response, aiohttp.ClientResponse],
        stream_callback: Optional[Union[Callable[[str], None], Callable[[str], Awaitable[None]]]],
        is_async: bool = False
    ) -> Dict[str, Any]:
        """Handle streaming response from either sync or async client."""
        accumulated_response = {
            "id": None,
            "model": None,
            "object": "chat.completion",
            "created": None,
            "choices": [{
                "index": 0,
                "finish_reason": None,
                "message": {"role": "assistant", "content": ""},
                "delta": {"role": "assistant", "content": ""}
            }],
            "usage": None,
            "citations": []  # Move citations to root level
        }

        is_async_callback = stream_callback and asyncio.iscoroutinefunction(stream_callback)

        async def process_line(line: str) -> None:
            line = line.strip()
            if not line.startswith("data: "):
                return
            
            data = line[6:]  # Remove "data: " prefix
            if data == "[DONE]":
                return

            try:
                chunk = json.loads(data)
                # Update metadata and citations
                accumulated_response.update({
                    k: v for k, v in chunk.items() 
                    if k in ["id", "model", "created", "citations"]
                })

                # Check for content delta
                if "choices" in chunk and chunk["choices"]:
                    choice = chunk["choices"][0]
                    delta = choice.get("delta", {})
                    
                    # Handle content
                    if "content" in delta:
                        token = delta["content"]
                        accumulated_response["choices"][0]["delta"]["content"] += token
                        accumulated_response["choices"][0]["message"]["content"] += token
                        
                        # Stream token
                        if stream_callback:
                            try:
                                if is_async_callback:
                                    await stream_callback(token)
                                else:
                                    stream_callback(token)
                            except Exception:
                                pass

                    # Handle completion
                    if "finish_reason" in choice:
                        accumulated_response["choices"][0]["finish_reason"] = choice["finish_reason"]

                # Handle usage stats
                if "usage" in chunk:
                    accumulated_response["usage"] = chunk["usage"]

            except json.JSONDecodeError:
                pass

        if is_async:
            # Handle async response
            async for line in response.content:
                await process_line(line.decode('utf-8'))
        else:
            # Handle sync response
            for line in response.iter_lines():
                if line:
                    await process_line(line.decode('utf-8'))

        return accumulated_response

    @backoff.on_exception(
        backoff.expo,
        (requests.exceptions.RequestException, aiohttp.ClientError, PerplexityError),
        max_tries=MAX_RETRIES + 1,  # +1 because first try counts
        max_time=30,  # Maximum total time to try
        base=2,  # Use exponential backoff
        logger=None  # Let app handle logging
    )
    async def _make_request(
        self,
        request: PerplexityRequest,
        stream_callback: Optional[Union[Callable[[str], None], Callable[[str], Awaitable[None]]]] = None,
        is_async: bool = False,
        attempt: int = 0
    ) -> Dict[str, Any]:
        """Make either sync or async request to Perplexity API with retries."""
        try:
            # Convert request to dict and handle None values
            payload = request.model_dump(exclude_none=True)
            
            # Remove search_recency_filter if it's None
            if "search_recency_filter" in payload and payload["search_recency_filter"] is None:
                del payload["search_recency_filter"]

            if is_async:
                # Async request using aiohttp
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        self.BASE_URL,
                        json=payload,
                        headers=self._get_headers(),
                        timeout=aiohttp.ClientTimeout(total=60)
                    ) as response:
                        if not response.ok:
                            error_body = await response.text()
                            raise aiohttp.ClientError(f"{response.status}, message='{response.reason}', body='{error_body}', url='{response.url}'")
                        if request.stream:
                            return await self._stream_response(response, stream_callback, is_async=True)
                        else:
                            return await response.json()
            else:
                # Sync request using requests
                response = requests.post(
                    self.BASE_URL,
                    json=payload,
                    headers=self._get_headers(),
                    stream=request.stream,
                    timeout=60
                )
                if not response.ok:
                    error_body = response.text
                    raise requests.RequestException(f"{response.status_code}, message='{response.reason}', body='{error_body}', url='{response.url}'")
                
                if request.stream:
                    return await self._stream_response(response, stream_callback, is_async=False)
                else:
                    return response.json()

        except (requests.exceptions.RequestException, aiohttp.ClientError) as e:
            raise

    def chat_completion(
        self, 
        request: PerplexityRequest, 
        stream_callback: Optional[Union[Callable[[str], None], Callable[[str], Awaitable[None]]]] = None
    ) -> Dict[str, Any]:
        """Send a chat completion request to the Perplexity API."""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(
                    self._make_request(request, stream_callback, is_async=False)
                )
            finally:
                loop.close()
        except Exception as e:
            raise requests.RequestException(f"Error in Perplexity request: {str(e)}") from e

    async def async_chat_completion(
        self, 
        request: PerplexityRequest,
        stream_callback: Optional[Callable[[str], Awaitable[None]]] = None
    ) -> Dict[str, Any]:
        """Send an asynchronous chat completion request to the Perplexity API."""
        try:
            return await self._make_request(request, stream_callback, is_async=True)
        except Exception as e:
            raise aiohttp.ClientError(f"Error in Perplexity request: {str(e)}") from e

if __name__ == "__main__":
    import asyncio
    import sys
    from typing import Optional

    async def stream_handler(token: str) -> None:
        """Handle streaming tokens by printing them to stdout."""
        print(token, end="", flush=True)

    async def test_perplexity_api() -> None:
        """Test the Perplexity API with a sample query."""
        # Get API key from environment
        api_key = os.environ.get("PERPLEXITY_API_KEY")
        if not api_key:
            print("Error: PERPLEXITY_API_KEY environment variable not set", file=sys.stderr)
            sys.exit(1)

        # Initialize client
        client = PerplexityClient(api_key=api_key)

        # Create request matching example format exactly
        request = PerplexityRequest(
            model="sonar",
            messages=[
                Message(role="system", content="Be precise and concise."),
                Message(role="user", content="Who is Kamiwaza.ai?")
            ],
            max_tokens=2048,
            temperature=0.5,
            top_p=0.95,
            search_domain_filter=["<any>"],
            return_images=False,
            return_related_questions=False,
            top_k=0,
            stream=False,  # Match their example exactly
            presence_penalty=0,
            frequency_penalty=1,
            web_search_options={"search_context_size": "high"}
        )

        try:
            print("\nTesting non-streaming response:")
            print("-" * 80)
            response = await client.async_chat_completion(request)
            print(response["choices"][0]["message"]["content"])
            print("\nCitations:")
            print("-" * 80)
            citations = response.get("citations", [])
            for i, citation in enumerate(citations, 1):
                print(f"{i}. {citation}")

            print("\n\nTesting streaming response:")
            print("-" * 80)
            request.stream = True
            response = await client.async_chat_completion(
                request,
                stream_callback=stream_handler
            )
            print("\n\nCitations:")
            print("-" * 80)
            citations = response.get("citations", [])
            for i, citation in enumerate(citations, 1):
                print(f"{i}. {citation}")

        except Exception as e:
            print(f"Error testing Perplexity API: {e}", file=sys.stderr)
            sys.exit(1)

    # Run the async test
    asyncio.run(test_perplexity_api())