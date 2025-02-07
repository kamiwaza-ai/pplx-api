import os
from typing import List, Dict, Optional, Union, Any, Callable, Awaitable
import requests
from pydantic import BaseModel, Field, field_validator
from pydantic_core import PydanticCustomError
import json
import asyncio
from functools import partial
import aiohttp

class Message(BaseModel):
    role: str
    content: str

class PerplexityRequest(BaseModel):
    model: str = Field(default="llama-3.1-sonar-large-128k-online")
    messages: List[Message]
    max_tokens: Optional[int] = None
    temperature: float = Field(default=0.2, ge=0, lt=2)
    top_p: float = Field(default=0.9, ge=0, le=1)
    return_citations: bool = Field(default=False)
    search_domain_filter: Optional[List[str]] = Field(default=None, max_length=3)
    return_images: bool = Field(default=False)
    return_related_questions: bool = Field(default=False)
    search_recency_filter: Optional[str] = Field(default=None)
    top_k: int = Field(default=0, ge=0, le=2048)
    stream: bool = Field(default=True)
    presence_penalty: float = Field(default=0, ge=-2, le=2)
    frequency_penalty: float = Field(default=1, ge=0)

    @field_validator('search_recency_filter')
    def validate_search_recency_filter(cls, v):
        if v is not None and v not in ['month', 'week', 'day', 'hour']:
            raise PydanticCustomError(
                'invalid_search_recency_filter',
                'search_recency_filter must be one of: month, week, day, hour'
            )
        return v

    model_config = {
        'extra': 'forbid'
    }

class PerplexityClient:
    """
    A client for interacting with the Perplexity AI API.
    """

    BASE_URL = "https://api.perplexity.ai/chat/completions"

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Perplexity client.

        :param api_key: Optional API key. If not provided, it will be read from the PERPLEXITY_API_KEY environment variable.
        """
        self.api_key = api_key or os.environ.get("PERPLEXITY_API_KEY")
        if not self.api_key:
            raise ValueError("API key must be provided either as a parameter or through the PERPLEXITY_API_KEY environment variable.")

    def _get_headers(self) -> Dict[str, str]:
        """
        Get the headers for the API request.

        :return: A dictionary of headers.
        """
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def _handle_async_callback(self, callback: Callable[[str], Awaitable[None]], content: str) -> None:
        """
        Safely handle async callbacks by creating a task in the current loop
        """
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(callback(content))
        except RuntimeError:
            # No event loop - this shouldn't happen in our use case
            # but we'll handle it gracefully
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(callback(content))
            finally:
                loop.close()

    def chat_completion(
        self, 
        request: PerplexityRequest, 
        stream_callback: Optional[Union[Callable[[str], None], Callable[[str], Awaitable[None]]]] = None
    ) -> Dict[str, Any]:
        """
        Send a chat completion request to the Perplexity API.

        :param request: A PerplexityRequest object containing the request parameters.
        :param stream_callback: Optional callback function (sync or async) to handle streaming responses.
        :return: The API response as a dictionary.
        :raises requests.RequestException: If there's an error making the request to the Perplexity API.
        """
        try:
            try:
                response = requests.post(
                    self.BASE_URL,
                    json=request.model_dump(exclude_none=True),
                    headers=self._get_headers(),
                    stream=request.stream,
                    timeout=60
                )
            except requests.Timeout:
                raise requests.RequestException("Request to Perplexity API timed out after 60 seconds") from None
            response.raise_for_status()
            
            if request.stream:
                return self._handle_stream_response(response, stream_callback)
            else:
                return response.json()
        except requests.RequestException as e:
            # Re-raise the exception with additional context
            raise requests.RequestException(f"Error making request to Perplexity API: {str(e)}") from e

    def _handle_stream_response(
        self, 
        response: requests.Response, 
        stream_callback: Optional[Union[Callable[[str], None], Callable[[str], Awaitable[None]]]] = None
    ) -> Dict[str, Any]:
        """
        Handle streaming response from the Perplexity API.

        :param response: The streaming response object.
        :param stream_callback: Optional callback function (sync or async) to handle streaming responses.
        :return: A dictionary containing the accumulated response data.
        """
        accumulated_response = {
            "id": None,
            "model": None,
            "object": "chat.completion",
            "created": None,
            "choices": [{
                "index": 0,
                "finish_reason": None,
                "message": {
                    "role": "assistant",
                    "content": ""
                },
                "delta": {
                    "role": "assistant",
                    "content": ""
                }
            }],
            "usage": {
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None
            }
        }

        is_async_callback = stream_callback and asyncio.iscoroutinefunction(stream_callback)

        for line in response.iter_lines():
            if line:
                event_data = line.decode('utf-8').strip()
                if event_data.startswith("data: "):
                    data = event_data[6:]  # Remove "data: " prefix
                    if data != "[DONE]":
                        try:
                            chunk = json.loads(data)
                            # Update accumulated response with chunk data
                            accumulated_response.update({k: v for k, v in chunk.items() if k not in ["choices", "usage"]})
                            content = chunk["choices"][0]["delta"].get("content", "")
                            accumulated_response["choices"][0]["delta"]["content"] += content
                            accumulated_response["choices"][0]["message"]["content"] += content
                            if "finish_reason" in chunk["choices"][0]:
                                accumulated_response["choices"][0]["finish_reason"] = chunk["choices"][0]["finish_reason"]
                            if "usage" in chunk:
                                accumulated_response["usage"] = chunk["usage"]
                            
                            # Call the stream callback if provided
                            if stream_callback and content:
                                if is_async_callback:
                                    self._handle_async_callback(stream_callback, content)
                                else:
                                    stream_callback(content)
                        except json.JSONDecodeError:
                            print(f"Failed to parse JSON from stream: {data}")
        
        return accumulated_response

    async def async_chat_completion(
        self, 
        request: PerplexityRequest,
        stream_callback: Optional[Callable[[str], Awaitable[None]]] = None
    ) -> Dict[str, Any]:
        """
        Send an asynchronous chat completion request to the Perplexity API.

        :param request: A PerplexityRequest object containing the request parameters.
        :param stream_callback: Optional async callback function to handle streaming responses.
        :return: The API response as a dictionary.
        :raises aiohttp.ClientError: If there's an error making the request to the Perplexity API.
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.BASE_URL,
                    json=request.model_dump(exclude_none=True),
                    headers=self._get_headers(),
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    response.raise_for_status()
                    
                    if request.stream:
                        return await self._handle_async_stream_response(response, stream_callback)
                    else:
                        return await response.json()
                        
        except aiohttp.ClientError as e:
            raise aiohttp.ClientError(f"Error making request to Perplexity API: {str(e)}") from e

    async def _handle_async_stream_response(
        self,
        response: aiohttp.ClientResponse,
        stream_callback: Optional[Callable[[str], Awaitable[None]]] = None
    ) -> Dict[str, Any]:
        """
        Handle async streaming response from the Perplexity API.

        :param response: The streaming response object.
        :param stream_callback: Optional async callback function to handle streaming responses.
        :return: A dictionary containing the accumulated response data.
        """
        accumulated_response = {
            "id": None,
            "model": None,
            "object": "chat.completion",
            "created": None,
            "choices": [{
                "index": 0,
                "finish_reason": None,
                "message": {
                    "role": "assistant",
                    "content": ""
                },
                "delta": {
                    "role": "assistant",
                    "content": ""
                }
            }],
            "usage": {
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None
            }
        }

        async for line in response.content:
            if line:
                event_data = line.decode('utf-8').strip()
                if event_data.startswith("data: "):
                    data = event_data[6:]  # Remove "data: " prefix
                    if data != "[DONE]":
                        try:
                            chunk = json.loads(data)
                            # Update accumulated response with chunk data
                            accumulated_response.update({k: v for k, v in chunk.items() if k not in ["choices", "usage"]})
                            content = chunk["choices"][0]["delta"].get("content", "")
                            accumulated_response["choices"][0]["delta"]["content"] += content
                            accumulated_response["choices"][0]["message"]["content"] += content
                            if "finish_reason" in chunk["choices"][0]:
                                accumulated_response["choices"][0]["finish_reason"] = chunk["choices"][0]["finish_reason"]
                            if "usage" in chunk:
                                accumulated_response["usage"] = chunk["usage"]
                            
                            # Call the stream callback if provided
                            if stream_callback and content:
                                await stream_callback(content)
                        except json.JSONDecodeError:
                            print(f"Failed to parse JSON from stream: {data}")
        
        return accumulated_response

def print_stream(content: str):
    """
    Print the streaming response token by token.
    """
    print(content, end='', flush=True)

# Example usage:
if __name__ == "__main__":
    async def async_print_stream(content: str):
        """
        Async print the streaming response token by token.
        """
        print(content, end='', flush=True)

    async def main():
        client = PerplexityClient()
        request = PerplexityRequest(
            messages=[Message(role="user", content="What is Kamiwaza.AI?")]
        )
        print("Streaming response:")
        response = await client.async_chat_completion(request, stream_callback=async_print_stream)
        print("\n\nFull response:")
        print(json.dumps(response, indent=2))
        print("\n\n")
        print(response["choices"][0]["message"]["content"])

    asyncio.run(main())