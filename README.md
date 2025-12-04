# AnyLLM Monorepo

A Python monorepo containing the following packages:

## 📦 Packages

### 1. AnyLLM
Unified LLM client with compatible OpenAI and g4f interfaces.

**Features**:
- 🔄 Compatible with OpenAI Client and g4f Client APIs
- 🌐 Dual backend support (OpenAI, GPT4Free)
- 📝 Unified API calling approach
- 🔌 Easy to extend and integrate

Documentation: [packages/anyllm/README.md](packages/anyllm/README.md)

### 2. G4FAdmin
GPT4Free Provider and Model management tool to quickly find working provider/model combinations.

**Features**:
- 🔍 Provider scanning and recommendation
- 📋 Model listing and searching
- 🔐 Authentication detection
- ✅ Real API testing
- 🚀 Batch testing and export

Documentation: [packages/g4fadmin/README.md](packages/g4fadmin/README.md)

## 🏗️ Project Structure

```
AnyLLM/
├── README.md                 # This file
├── .gitignore               # Git ignore rules
├── config/                   # Shared configurations
│   ├── models_to_providers.json
│   └── vllm-Qwen3-0.6B.yaml
└── packages/                 # All packages
    ├── anyllm/              # AnyLLM package
    │   ├── pyproject.toml
    │   ├── README.md
    │   └── anyllm/
    │       ├── __init__.py
    │       ├── client.py
    │       └── result.py
    └── g4fadmin/            # G4FAdmin package
        ├── pyproject.toml
        ├── README.md
        └── g4fadmin/
            ├── __init__.py
            ├── admin.py
            ├── cli.py
            └── config.py
```

## 🚀 Quick Start

### Installation

```bash
# Install anyllm
cd packages/anyllm
pip install -e ".[all]"

# Install g4fadmin
cd ../g4fadmin
pip install -e .
```

### Usage Examples

#### AnyLLM
```python
from anyllm import Client

# Using OpenAI
client = Client(api_key="sk-xxx", model="gpt-4")
response = client.chat.completions.create(
    messages=[{"role": "user", "content": "Hello"}]
)

# Using g4f
client = Client(provider="DeepInfra")
response = client.chat.completions.create(
    model="auto",
    messages=[{"role": "user", "content": "Hello"}]
)
```

#### G4FAdmin
```python
from g4fadmin import G4FAdmin

admin = G4FAdmin()
admin.print_summary()
providers = admin.get_recommended_providers(5)
success, resp, resp_time = admin.test_provider("DeepInfra", model_name="gpt-4", test_prompt="Hello")
```

Or use CLI:
```bash
# View summary
g4fadmin

# Test a provider
g4fadmin --test DeepInfra --model gpt-4

# Find providers for a model
g4fadmin --find gpt-4

# Probe all working combinations
g4fadmin --probe
```

## 📄 License

MIT License
