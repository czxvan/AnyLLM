"""
G4FAdmin Configuration

Contains blacklist providers and authentication keyword configurations
"""

# Provider blacklist - these providers will be automatically filtered
BLACKLIST_PROVIDERS = [
    # Providers requiring .HAR files
    "Copilot",
    "CopilotAccount",
    "OpenaiAccount",
    "OpenaiChat",
    
    # Providers requiring browser_cookie3
    "GithubCopilot",
    
    # Providers requiring special dependencies
    "LMArena",  # Requires nodriver and platformdirs
    
    # Providers requiring special Cookies
    "Gemini",  # Requires __Secure-1PSID cookie
    
    # Special adapters (not real providers)
    "AnyProvider",
]

# Known stable providers (prioritized for recommendation)
# Note: This list is based on actual test results and needs regular updates
KNOWN_STABLE_PROVIDERS = [
    "DeepInfra",      # ✅ Tested working - supports multiple open-source models
    "PollinationsAI", # ✅ Tested working - open-source friendly
    # The following providers may need specific conditions or be slower, but usually work
    "HuggingFace",    # HuggingFace official
    "Qwen",           # Qwen - Alibaba Cloud service
]

# Authentication type keyword mapping
# Used to identify authentication types from error messages
AUTH_KEYWORDS = {
    "api_key": ["api key", "api_key", "apikey", "api-key"],
    "cookie": ["cookie", "__secure", "session"],
    "token": ["token", "bearer", "authorization"],  
    "har_file": [".har", "har file", "browser_cookie"],
    "account": ["login", "account", "credentials", "username", "password"],
}
