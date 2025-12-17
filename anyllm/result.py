from types import GeneratorType
from typing import AsyncIterator, Union
from pydantic import BaseModel


def to_result(resp: Union[dict, BaseModel, GeneratorType]) -> dict:
    """
    Convert model response to dictionary format.
    
    Args:
        resp: Response object (dict, BaseModel, or Generator for streaming)
        
    Returns:
        Unified result dictionary with 'content', 'usage', and 'finish_reason'
        
    Examples:
        # Non-streaming
        response = client.chat.completions.create(messages="Hello")
        result = to_result(response)
        
        # Streaming (sync)
        response = client.chat.completions.create(messages="Hello", stream=True)
        result = to_result(response)  # Consumes the entire stream
        
        # Async - await first, then convert
        response = await async_client.chat.completions.create(messages="Hello")
        result = to_result(response)
    """
    # Check for async iterator - provide helpful error
    if hasattr(resp, '__aiter__'):
        raise TypeError(
            "Async iterator detected. Please use ato_result() instead:\n"
            "  result = await ato_result(response)"
        )
    
    # Handle streaming (sync generator)
    if isinstance(resp, GeneratorType):
        result = {
            "content": "",
            "usage": {
                'total_tokens': 0,
                'completion_tokens': 0,
                'prompt_tokens': 0,
            },
            'finish_reason': 'unknown',
        }

        chunk_result = None
        for chunk in resp:
            chunk_result = to_result(chunk)
            result['content'] += chunk_result['content']

        # Update usage from last chunk if available
        if chunk_result and chunk_result['usage']['total_tokens'] >= 0:
            result['usage'] = chunk_result['usage']
            result['finish_reason'] = chunk_result['finish_reason']
        return result

    # Convert BaseModel to dict
    if isinstance(resp, BaseModel):
        resp = resp.model_dump()
    elif isinstance(resp, dict):
        resp = resp
    else:
        raise ValueError(f"Unsupported response type: {type(resp)}")
        
    
    choice = resp['choices'][0]
    message = choice.get('message') or choice.get('delta')
    if message is None:
        raise ValueError("No message or delta found in choices")
    if not isinstance(message, dict):
        raise ValueError("Message or delta is not a dictionary")

    if 'reasoning' in message and message['reasoning']:
        content = str(message['reasoning']) + message.get('content', '')
    else:
        content = message.get('content', '') 

    usage = resp.get('usage', choice.get('usage'))
    if usage is None \
         or not isinstance(usage, dict) \
         or 'total_tokens' not in usage \
         or 'completion_tokens' not in usage \
         or 'prompt_tokens' not in usage:
        usage = {
            'total_tokens': -1,
            'completion_tokens': -1,
            'prompt_tokens': -1,
        }
    
    finish_reason = choice.get('finish_reason', 'unknown')

    res = {
        "content": content,
        "usage": {
            'total_tokens': usage['total_tokens'],
            'completion_tokens': usage['completion_tokens'],
            'prompt_tokens': usage['prompt_tokens'],
        },
        'finish_reason': finish_reason,
    }
    return res


async def ato_result(resp: Union[dict, BaseModel, AsyncIterator]) -> dict:
    """
    Convert async model response to dictionary format.
    
    Handles async iterators (streaming) automatically by consuming the entire stream.
    
    Note: Streaming responses are NOT suitable for parallel processing with asyncio.gather()
    because each stream must be consumed sequentially. For parallel requests, use 
    non-streaming mode (stream=False).
    
    Args:
        resp: Async response object (dict, BaseModel, or AsyncIterator for streaming)
        
    Returns:
        Unified result dictionary with 'content', 'usage', and 'finish_reason'
        
    Examples:
        # Non-streaming (suitable for parallel processing)
        response = await client.chat.completions.create(messages="Hello")
        result = await ato_result(response)
        
        # Streaming - automatically consumes the entire stream (sequential only)
        response = await client.chat.completions.create(messages="Hello", stream=True)
        result = await ato_result(response)  # Must wait for entire stream
        
        # For parallel processing, use non-streaming:
        tasks = [client.chat.completions.create(messages=f"Q{i}") for i in range(10)]
        responses = await asyncio.gather(*tasks)
        results = [to_result(r) for r in responses]
    """
    # Check if it's an async iterator (streaming)
    if hasattr(resp, '__aiter__'):
        result = {
            "content": "",
            "usage": {
                'total_tokens': 0,
                'completion_tokens': 0,
                'prompt_tokens': 0,
            },
            'finish_reason': 'unknown',
        }

        chunk_result = None
        async for chunk in resp:
            chunk_result = to_result(chunk)
            result['content'] += chunk_result['content']

        # Update usage from last chunk if available
        if chunk_result and chunk_result['usage']['total_tokens'] >= 0:
            result['usage'] = chunk_result['usage']
            result['finish_reason'] = chunk_result['finish_reason']
        return result

    # For non-stream responses, use the sync version
    return to_result(resp)
