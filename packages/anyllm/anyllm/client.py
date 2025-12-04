from typing import Optional, Union, List, Dict, Any
import os


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
        # Normalize messages
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        
        # Use provided model or client's default model
        if model is None:
            model = self.client.model
        
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
        
        # Call appropriate implementation based on backend type
        return self.client._create_completion(params)


class Chat:
    """Chat interface compatible with OpenAI and g4f chat API"""
    
    def __init__(self, client: 'Client'):
        self.completions = ChatCompletions(client)


class Client:
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
            elif base_url is not None:  # User explicitly specified base_url
                backend = "openai"
            else:
                backend = "gpt4free"  # Default to free backend
        
        self.backend = backend
        self.provider = provider
        
        # Initialize chat interface
        self.chat = Chat(self)
        
        # Lazy initialization of backend client
        self._backend_client = None
    
    def _get_backend_client(self):
        """Lazy initialize backend client"""
        if self._backend_client is not None:
            return self._backend_client
        
        if self.backend == "openai":
            # Use OpenAI SDK
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
            # Use g4f Client
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
            # Call OpenAI API
            return backend_client.chat.completions.create(**params)
        
        elif self.backend == "gpt4free":
            # Call g4f API
            if self.provider:
                params["provider"] = self.provider
            return backend_client.chat.completions.create(**params)
        
        else:
            raise ValueError(f"不支持的后端类型: {self.backend}")