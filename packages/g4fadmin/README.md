# G4FAdmin

GPT4Free Provider and Model management tool to quickly find which provider/model combinations in g4f actually work.

## ✨ Features

- 🔍 **Provider Scanning**: Automatically scan all available g4f providers
- 📋 **Model Listing**: Get all models and their supporting providers
- 🔐 **Authentication Detection**: Identify provider authentication methods through testing
- ✅ **Real Testing**: Test actual availability of provider/model combinations
- 🚀 **Batch Testing**: Concurrent testing of multiple combinations for efficiency
- 📊 **Data Export**: Export test results to JSON format
- 🎯 **Smart Recommendations**: Recommend available providers based on blacklist and stability
- 📝 **Detailed Information**: Display provider features and model support

## 📦 Installation

```bash
pip install -e .
```

## 🚀 Quick Start

### Using as Python Library

```python
from g4fadmin import G4FAdmin

# Initialize
admin = G4FAdmin()

# View summary
admin.print_summary()

# Get recommended providers
providers = admin.get_recommended_providers(top_n=5)
for p in providers:
    print(f"{p.name}: {len(p.models)} models")

# Test specific provider and model
success, response, response_time = admin.test_provider(
    provider_name="DeepInfra",
    model_name="meta-llama/Llama-3.3-70B-Instruct",
    test_prompt="Hello"
)

if success:
    print(f"✅ Test successful ({response_time:.2f}s)")
    print(response)
else:
    print(f"❌ Test failed: {response}")

# Batch testing
results = admin.batch_test_providers(
    provider_names=["DeepInfra", "PollinationsAI"],
    test_prompt="Hello",
    timeout=15
)

# Export results
admin.export_test_results(results, "test_results.json")
```

### Using as Command Line Tool

```bash
# List all providers
g4fadmin --list-providers

# List only working providers
g4fadmin --list-providers --working-only

# List all models
g4fadmin --list-models

# Find providers supporting specific model
g4fadmin --find gpt-4

# Find and test providers for a model
g4fadmin --find gpt-4 --test-find

# View summary
g4fadmin

# View summary with real tests
g4fadmin --real-test

# Get recommended providers (shown in summary)
g4fadmin

# Test specific provider
g4fadmin --test DeepInfra --model gpt-4

# Batch testing (uses recommended providers)
g4fadmin --batch-test --prompt "1+1=?"

# Probe all working combinations and export
g4fadmin --probe --output-dir output

# Export provider and model information
g4fadmin --export --output-dir output
```

## 📚 API Reference

### G4FAdmin Class

#### Initialization

```python
admin = G4FAdmin()
```

#### Main Methods

##### get_all_providers()
Get list of all providers.

**Returns**: `List[ProviderInfo]`

##### get_working_providers()
Get providers marked as working.

**Returns**: `List[ProviderInfo]`

##### get_recommended_providers(top_n=10)
Get recommended providers (filtered blacklist, prioritize known stable ones).

**Parameters**:
- `top_n` (int): Number top_n

**Returns**: `List[ProviderInfo]`

##### get_all_models()
Get all available models.

**Returns**: `List[ModelInfo]`

##### find_providers_for_model(model_name)
Find providers supporting specific model.

**Parameters**:
- `model_name` (str): Model name

**Returns**: `List[ProviderInfo]`

##### test_provider(provider_name, model_name=None, test_prompt="Hello", timeout=15, verbose=False)
Test availability of specific provider and model.

**Parameters**:
- `provider_name` (str): Provider name
- `model_name` (str, optional): Model name, default None (auto-select)
- `test_prompt` (str): Test prompt, default "Hello"
- `timeout` (int): Timeout in seconds, default 15
- `verbose` (bool): Verbose output, default False

**Returns**: `Tuple[bool, Optional[str], Optional[float]]`
- `bool`: Whether successful
- `Optional[str]`: Response content or error message
- `Optional[float]`: Response time in seconds

##### batch_test_providers(provider_names=None, test_prompt="1+1=?", timeout=15)
Batch test multiple providers.

**Parameters**:
- `provider_names` (Optional[List[str]]): Provider name list, default None (uses recommended providers)
- `test_prompt` (str): Test prompt, default "1+1=?"
- `timeout` (int): Timeout in seconds, default 15

**Returns**: `Dict[str, Tuple[bool, Optional[str], Optional[float]]]`
Dictionary mapping provider names to test results (success, response, response_time)

##### export_test_results(test_results, filepath)
Export test results to JSON file.

**Parameters**:
- `test_results` (List[TestResult]): Test result list
- `filepath` (str): Output file path

##### print_summary()
Print summary of providers and models.

### Data Classes

#### ProviderInfo
Provider information.

**Fields**:
- `name` (str): Provider name
- `working` (bool): Whether working
- `supports_stream` (bool): Whether supports streaming
- `supports_message_history` (bool): Whether supports message history
- `supports_system_message` (bool): Whether supports system message
- `models` (List[str]): List of supported models
- `auth_type` (AuthType): Authentication type, default AuthType.NONE
- `auth_required` (bool): Whether requires authentication, default False

#### ModelInfo
Model information.

**Fields**:
- `name` (str): Model name
- `providers` (List[str]): List of providers supporting this model

#### TestResult
Test result.

**Fields**:
- `provider` (str): Provider name
- `model` (str): Model name
- `success` (bool): Whether successful
- `response` (Optional[str]): Response content or error message
- `response_time` (Optional[float]): Response time in seconds
- `auth_type` (AuthType): Detected authentication type, default AuthType.NONE
- `timestamp` (datetime): Test timestamp

#### AuthType
Authentication type enum.

**Values**:
- `NONE`: No authentication required
- `API_KEY`: Requires API Key
- `COOKIE`: Requires Cookie
- `TOKEN`: Requires Token
- `HAR_FILE`: Requires HAR file
- `ACCOUNT`: Requires account credentials
- `UNKNOWN`: Unknown type

## 🔧 Configuration

### Provider Blacklist

Configure blacklist providers in `g4fadmin/config.py` (these usually require special dependencies or configuration):

```python
BLACKLIST_PROVIDERS = [
    "Copilot",
    "OpenaiAccount",
    "GithubCopilot",
    # ...
]
```

### Known Stable Providers

Configure known stable providers which will be prioritized for recommendation:

```python
KNOWN_STABLE_PROVIDERS = [
    "DeepInfra",
    "PollinationsAI",
    "HuggingFace",
    "Qwen",
]
```

## 📊 Usage Examples

### Example 1: Find Actually Working Providers

```python
from g4fadmin import G4FAdmin

admin = G4FAdmin()

# Get recommended providers
providers = admin.get_recommended_providers(top_n=5)

# Batch test (returns Dict, not List)
results = admin.batch_test_providers(
    provider_names=[p.name for p in providers],
    test_prompt="Hello",
    timeout=15
)

# Filter successful ones
successful = [(name, info) for name, (success, _, _) in results.items() if success]
print(f"✅ Working providers: {len(successful)}/{len(results)}")

for name, (success, response, response_time) in successful:
    print(f"  - {name} ({response_time:.2f}s)")
```

### Example 2: Find Providers Supporting Specific Model

```python
from g4fadmin import G4FAdmin

admin = G4FAdmin()

# Find providers supporting gpt-4 (returns List[str])
provider_names = admin.find_providers_for_model("gpt-4")

print(f"Providers supporting gpt-4 ({len(provider_names)}):")
for name in provider_names:
    print(f"  - {name}")
```

### Example 3: Test and Export Results

```python
from g4fadmin import G4FAdmin

admin = G4FAdmin()

# Batch test all recommended providers
providers = admin.get_recommended_providers()
results = admin.batch_test_providers(
    provider_names=[p.name for p in providers],
    test_prompt="Hello"
)

# Export to JSON (Note: batch_test_providers returns Dict, not List[TestResult])
# For export, you may need to use test_all_combinations instead
test_results = admin.test_all_combinations(test_prompt="Hello")
admin.export_test_results(test_results, "test_results.json")
print("✅ Results exported to test_results.json")
```

## 📄 License

MIT License
