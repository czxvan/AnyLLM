# AnyLLM

A unified LLM client with compatible OpenAI and g4f interfaces.

## Features

- 🔄 **Unified Interface**: Compatible with both OpenAI Client and g4f Client API styles
- ⚡ **Async Support**: Full async/await support for high-performance applications
- 🔌 **Multiple Backends**: Support for OpenAI, g4f (GPT4Free), and more
- 🎯 **Auto Detection**: Automatically selects the appropriate backend based on parameters
- 📦 **Lightweight Design**: Minimal core dependencies with flexible optional dependencies
- 🚀 **Easy to Use**: Simple API for quick start

## Installation

### Basic Installation

```bash
pip install anyllm
```

### Optional Dependencies

Install dependencies for different backends as needed:

```bash
# Install OpenAI support
pip install anyllm[openai]

# Install g4f support
pip install anyllm[g4f]

# Install all optional dependencies
pip install anyllm[all]
```

## Quick Start

### Synchronous Usage

#### Using OpenAI Backend

```python
from anyllm import Client

# Method 1: Using API key
client = Client(
    api_key="sk-xxx",
    base_url="https://api.openai.com/v1",
    model="gpt-4"
)

# Method 2: Using environment variables (auto-reads OPENAI_API_KEY)
client = Client(model="gpt-4")

# Send request
response = client.chat.completions.create(
    messages=[{"role": "user", "content": "Hello!"}]
)

print(response.choices[0].message.content)
```

#### Using g4f Backend (Free)

```python
from anyllm import Client

# Using g4f provider
client = Client(
    provider="Perplexity",  # or other g4f supported providers
    model="auto"
)

response = client.chat.completions.create(
    messages=[{"role": "user", "content": "Hello!"}]
)

print(response.choices[0].message.content)
```

#### Using Custom OpenAI-Compatible Services (e.g., vLLM)

```python
from anyllm import Client

# Connect to local or remote vLLM service
client = Client(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",  # vLLM typically doesn't require a real API key
    model="Qwen/Qwen2.5-3B-Instruct"
)

response = client.chat.completions.create(
    messages=[{"role": "user", "content": "Hello!"}]
)

print(response.choices[0].message.content)
```

### Asynchronous Usage

AnyLLM provides full async/await support for high-performance applications.

#### Basic Async Example

```python
import asyncio
from anyllm import AsyncClient, ato_result

async def main():
    # Create async client
    client = AsyncClient(
        api_key="sk-xxx",
        model="gpt-4"
    )
    
    # Async request
    response = await client.chat.completions.create(
        messages=[{"role": "user", "content": "Hello!"}]
    )
    
    # Convert to result
    result = await ato_result(response)
    print(result['content'])

asyncio.run(main())
```

#### Async Streaming

```python
import asyncio
from anyllm import AsyncClient

async def main():
    client = AsyncClient(model="gpt-4")
    
    response = await client.chat.completions.create(
        messages=[{"role": "user", "content": "Write a poem"}],
        stream=True
    )
    
    # Use async for to iterate over chunks
    async for chunk in response:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)

asyncio.run(main())
```

#### Concurrent Requests

```python
import asyncio
from anyllm import AsyncClient, ato_result

async def main():
    client = AsyncClient(api_key="sk-xxx")
    
    # Send multiple requests concurrently
    tasks = [
        client.chat.completions.create(messages=f"What is {topic}?")
        for topic in ["Python", "JavaScript", "Rust"]
    ]
    
    responses = await asyncio.gather(*tasks)
    
    for response in responses:
        result = await ato_result(response)
        print(result['content'])

asyncio.run(main())
```

#### Async Context Manager

```python
import asyncio
from anyllm import AsyncClient

async def main():
    async with AsyncClient(api_key="sk-xxx") as client:
        response = await client.chat.completions.create(
            messages="What is Python?"
        )
        print(response.choices[0].message.content)

asyncio.run(main())
```

## API Reference

### Synchronous API

#### Client Initialization

```python
Client(
    api_key: Optional[str] = None,        # API key
    base_url: Optional[str] = None,       # API base URL
    model: Optional[str] = None,          # Default model name
    backend: Optional[str] = None,        # Backend type ('openai', 'gpt4free')
    provider: Optional[str] = None,       # Provider name (for g4f backend)
    timeout: int = 60,                     # Request timeout in seconds
    proxy: Optional[str] = None,          # Proxy settings
    **kwargs                              # Other backend-specific parameters
)
```

### Asynchronous API

#### AsyncClient Initialization

```python
AsyncClient(
    api_key: Optional[str] = None,        # API key
    base_url: Optional[str] = None,       # API base URL
    model: Optional[str] = None,          # Default model name
    backend: Optional[str] = None,        # Backend type ('openai', 'gpt4free')
    provider: Optional[str] = None,       # Provider name (for g4f backend)
    timeout: int = 60,                     # Request timeout in seconds
    proxy: Optional[str] = None,          # Proxy settings
    **kwargs                              # Other backend-specific parameters
)
```

### Creating Chat Completions

Both sync and async clients use the same method signature:

```python
# Synchronous
client.chat.completions.create(...)

# Asynchronous
await async_client.chat.completions.create(...)
```

**Parameters:**
```python
messages: Union[List[Dict[str, str]], str],  # Message list or single string
model: Optional[str] = None,                  # Model name
stream: bool = False,                         # Whether to stream output
temperature: Optional[float] = None,          # Temperature parameter (0-2)
max_tokens: Optional[int] = None,             # Maximum tokens to generate
top_p: Optional[float] = None,                # Nucleus sampling parameter
frequency_penalty: Optional[float] = None,    # Frequency penalty
presence_penalty: Optional[float] = None,     # Presence penalty
stop: Optional[Union[str, List[str]]] = None, # Stop sequences
**kwargs                                      # Other parameters
```

### Result Conversion

#### Synchronous: `to_result`

Use the `to_result` function to convert responses to a unified dictionary format:

```python
from anyllm import Client, to_result

client = Client(model="gpt-4")
response = client.chat.completions.create(
    messages=[{"role": "user", "content": "Hello!"}]
)

# 转换为标准化的字典格式
result = to_result(response)
print(result)
# {
#     "content": "Response content",
#     "usage": {
#         "total_tokens": 100,
#         "completion_tokens": 50,
#         "prompt_tokens": 50
#     },
#     "finish_reason": "stop"
# }
```

#### Asynchronous: `ato_result`

Use the `ato_result` function to convert async responses:

```python
from anyllm import AsyncClient, ato_result
import asyncio

async def main():
    client = AsyncClient(model="gpt-4")
    response = await client.chat.completions.create(
        messages=[{"role": "user", "content": "Hello!"}]
    )
    
    # 转换为标准化的字典格式
    result = await ato_result(response)
    print(result)

asyncio.run(main())
```

## Advanced Usage

### Synchronous Streaming Output

```python
from anyllm import Client

client = Client(model="gpt-4")

stream = client.chat.completions.create(
    messages=[{"role": "user", "content": "Write a poem"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### Simplified Message Format

Supports passing a string directly as a user message:

```python
# The following two methods are equivalent
response = client.chat.completions.create(
    messages="Hello!"
)

response = client.chat.completions.create(
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### Custom Parameters

```python
response = client.chat.completions.create(
    messages=[{"role": "user", "content": "Tell me a joke"}],
    temperature=0.8,
    max_tokens=200,
    top_p=0.9,
    stop=["\n\n"]
)
```

## Backend Auto-Detection

AnyLLM automatically selects the appropriate backend based on the provided parameters:

- If `provider` is specified → Use g4f backend
- If `api_key` is specified → Use OpenAI backend
- If `base_url` is specified → Use OpenAI backend
- Otherwise → Default to g4f backend (free)

You can also explicitly specify the backend using the `backend` parameter:

```python
client = Client(backend="openai", api_key="sk-xxx")
client = Client(backend="gpt4free", provider="Perplexity")
```

## Environment Variables

AnyLLM supports reading configuration from environment variables:

- `OPENAI_API_KEY`: OpenAI API key
- `OPENAI_BASE_URL`: OpenAI API base URL (default: `https://api.openai.com/v1`)

```bash
export OPENAI_API_KEY="sk-xxx"
export OPENAI_BASE_URL="https://api.openai.com/v1"
```

```python
# No need to pass parameters, automatically reads from environment variables
client = Client(model="gpt-4")
```

## Contributing

Issues and Pull Requests are welcome!

## License

MIT License
