from typing import Optional, Union, List, Dict, Any
import os
from abc import ABC, abstractmethod


class BaseClient(ABC):
    """Base client with shared configuration and logic"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        backend: Optional[str] = None,
        provider: Optional[str] = None,
        timeout: int = 60,
        proxy: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize client
        
        Args:
            api_key: API key (OpenAI backend)
            base_url: API base URL (OpenAI-compatible backend)
            model: Default model name
            backend: Backend type ('openai', 'gpt4free', 'vllm', 'transformers')
            provider: Provider name (g4f backend)
            timeout: Request timeout
            proxy: Proxy settings
            **kwargs: Other backend-specific parameters
        """
        # Basic configuration
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.model = model or "gpt-3.5-turbo"
        self.timeout = timeout
        self.proxy = proxy
        self.kwargs = kwargs
        
        # Auto-infer backend type
        if backend is None:
            if provider is not None:
                backend = "gpt4free"
            elif api_key is not None:
                backend = "openai"
            elif base_url is not None:
                backend = "openai"
            else:
                backend = "gpt4free"
        
        self.backend = backend
        self.provider = provider
        
        # Lazy initialization of backend client
        self._backend_client = None
    
    def _normalize_params(
        self,
        messages: Union[List[Dict[str, str]], str],
        model: Optional[str] = None,
        stream: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        stop: Optional[Union[str, List[str]]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Normalize completion parameters"""
        # Normalize messages
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        
        # Use provided model or client's default model
        if model is None:
            model = self.model
        
        # Build parameters
        params = {
            "messages": messages,
            "model": model,
            "stream": stream,
        }
        
        # Add optional parameters
        if temperature is not None:
            params["temperature"] = temperature
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        if top_p is not None:
            params["top_p"] = top_p
        if frequency_penalty is not None:
            params["frequency_penalty"] = frequency_penalty
        if presence_penalty is not None:
            params["presence_penalty"] = presence_penalty
        if stop is not None:
            params["stop"] = stop
        
        # Merge other parameters
        params.update(kwargs)
        
        return params
    
    @abstractmethod
    def _get_backend_client(self):
        """Get or initialize backend client"""
        pass
    
    @abstractmethod
    def _create_completion(self, params: Dict[str, Any]):
        """Create completion using backend"""
        pass


# ==================== Sync Client ====================


class ChatCompletions:
    """Chat completions interface compatible with OpenAI and g4f chat.completions API"""
    
    def __init__(self, client: 'Client'):
        self.client = client
    
    def create(
        self,
        messages: Union[List[Dict[str, str]], str],
        model: Optional[str] = None,
        stream: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        stop: Optional[Union[str, List[str]]] = None,
        **kwargs
    ):
        """
        Create chat completion
        
        Args:
            messages: Message list or single string (converted to user message)
            model: Model name
            stream: Whether to stream output
            temperature: Temperature parameter (0-2)
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            frequency_penalty: Frequency penalty
            presence_penalty: Presence penalty
            stop: Stop sequences
            **kwargs: Other backend-specific parameters
        
        Returns:
            If stream=False: ChatCompletion object
            If stream=True: ChatCompletionChunk iterator
        """
        params = self.client._normalize_params(
            messages=messages,
            model=model,
            stream=stream,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop,
            **kwargs
        )
        return self.client._create_completion(params)


class Chat:
    """Chat interface compatible with OpenAI and g4f chat API"""
    
    def __init__(self, client: 'Client'):
        self.completions = ChatCompletions(client)


class Client(BaseClient):
    """
    Unified LLM client compatible with OpenAI Client and g4f Client interfaces
    
    Usage example:
        # Similar to OpenAI Client
        client = Client(api_key="sk-xxx", base_url="https://api.openai.com/v1")
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}]
        )
        
        # Similar to g4f Client
        client = Client(provider="Perplexity")
        response = client.chat.completions.create(
            model="auto",
            messages=[{"role": "user", "content": "Hello"}]
        )
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        backend: Optional[str] = None,
        provider: Optional[str] = None,
        timeout: int = 60,
        proxy: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            model=model,
            backend=backend,
            provider=provider,
            timeout=timeout,
            proxy=proxy,
            **kwargs
        )
        # Initialize chat interface
        self.chat = Chat(self)
    
    def _get_backend_client(self):
        """Lazy initialize backend client"""
        if self._backend_client is not None:
            return self._backend_client
        
        if self.backend == "openai":
            try:
                import openai
                self._backend_client = openai.OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                    timeout=self.timeout,
                    **self.kwargs
                )
            except ImportError:
                raise ImportError("Please install openai: pip install openai")
        
        elif self.backend == "gpt4free":
            try:
                from g4f.client import Client as G4FClient
                self._backend_client = G4FClient(
                    provider=self.provider,
                    proxy=self.proxy,
                    **self.kwargs
                )
            except ImportError:
                raise ImportError("Please install g4f: pip install g4f")
        
        else:
            raise ValueError(f"不支持的后端类型: {self.backend}")
        
        return self._backend_client
    
    def _create_completion(self, params: Dict[str, Any]):
        """
        Call backend to create completion
        
        Args:
            params: Completion parameters
        
        Returns:
            Completion response
        """
        backend_client = self._get_backend_client()
        
        if self.backend == "openai":
            return backend_client.chat.completions.create(**params)
        elif self.backend == "gpt4free":
            if self.provider:
                params["provider"] = self.provider
            return backend_client.chat.completions.create(**params)
        else:
            raise ValueError(f"不支持的后端类型: {self.backend}")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if self._backend_client is not None:
            if hasattr(self._backend_client, 'close'):
                self._backend_client.close()
        return False


# ==================== Async Client ====================


class AsyncChatCompletions:
    """Async chat completions interface compatible with OpenAI and g4f chat.completions API"""
    
    def __init__(self, client: 'AsyncClient'):
        self.client = client
    
    async def create(
        self,
        messages: Union[List[Dict[str, str]], str],
        model: Optional[str] = None,
        stream: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        stop: Optional[Union[str, List[str]]] = None,
        **kwargs
    ):
        """
        Create chat completion asynchronously
        
        Args:
            messages: Message list or single string (converted to user message)
            model: Model name
            stream: Whether to stream output
            temperature: Temperature parameter (0-2)
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            frequency_penalty: Frequency penalty
            presence_penalty: Presence penalty
            stop: Stop sequences
            **kwargs: Other backend-specific parameters
        
        Returns:
            If stream=False: ChatCompletion object
            If stream=True: AsyncIterator of ChatCompletionChunk
        """
        params = self.client._normalize_params(
            messages=messages,
            model=model,
            stream=stream,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop,
            **kwargs
        )
        return await self.client._create_completion(params)


class AsyncChat:
    """Async chat interface compatible with OpenAI and g4f chat API"""
    
    def __init__(self, client: 'AsyncClient'):
        self.completions = AsyncChatCompletions(client)


class AsyncClient(BaseClient):
    """
    Async unified LLM client compatible with OpenAI AsyncClient and g4f AsyncClient interfaces
    
    Usage example:
        # Similar to OpenAI AsyncClient
        client = AsyncClient(api_key="sk-xxx", base_url="https://api.openai.com/v1")
        response = await client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}]
        )
        
        # Similar to g4f AsyncClient
        client = AsyncClient(provider="Perplexity")
        response = await client.chat.completions.create(
            model="auto",
            messages=[{"role": "user", "content": "Hello"}]
        )
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        backend: Optional[str] = None,
        provider: Optional[str] = None,
        timeout: int = 60,
        proxy: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            model=model,
            backend=backend,
            provider=provider,
            timeout=timeout,
            proxy=proxy,
            **kwargs
        )
        # Initialize chat interface
        self.chat = AsyncChat(self)
    
    def _get_backend_client(self):
        """Lazy initialize async backend client"""
        if self._backend_client is not None:
            return self._backend_client
        
        if self.backend == "openai":
            try:
                import openai
                self._backend_client = openai.AsyncOpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                    timeout=self.timeout,
                    **self.kwargs
                )
            except ImportError:
                raise ImportError("Please install openai: pip install openai")
        
        elif self.backend == "gpt4free":
            try:
                from g4f.client import AsyncClient as G4FAsyncClient
                self._backend_client = G4FAsyncClient(
                    provider=self.provider,
                    proxy=self.proxy,
                    **self.kwargs
                )
            except ImportError:
                raise ImportError("Please install g4f: pip install g4f")
        
        else:
            raise ValueError(f"不支持的后端类型: {self.backend}")
        
        return self._backend_client
    
    async def _create_completion(self, params: Dict[str, Any]):
        """
        Call backend to create completion asynchronously
        
        Args:
            params: Completion parameters
        
        Returns:
            Completion response or async generator (if stream=True)
        
        Note:
            Different backends handle streaming differently:
            - OpenAI: Both stream and non-stream return awaitables
            - g4f: stream returns async_generator (no await), non-stream returns awaitable
        """
        backend_client = self._get_backend_client()
        is_stream = params.get("stream", False)
        
        if self.backend == "openai":
            # OpenAI: Both streaming and non-streaming can be awaited
            return await backend_client.chat.completions.create(**params)
        elif self.backend == "gpt4free":
            # Add provider to params if specified
            if self.provider:
                params["provider"] = self.provider
            
            result = backend_client.chat.completions.create(**params)
            
            # g4f has different behavior: streaming returns async generator, non-streaming returns coroutine
            return result if is_stream else await result
        else:
            raise ValueError(f"不支持的后端类型: {self.backend}")
    
    async def __aenter__(self):
        """Async context manager entry"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self._backend_client is not None:
            if hasattr(self._backend_client, 'close'):
                await self._backend_client.close()
        return False
