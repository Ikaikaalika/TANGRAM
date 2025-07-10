# Google Gemini Integration for TANGRAM

TANGRAM now supports Google Gemini API as an alternative to local LLM inference. This provides cloud-based AI capabilities with high performance and reliability.

## Features

- **Multiple LLM Backends**: Local Ollama (DeepSeek) + Google Gemini
- **Automatic Fallback**: Falls back to cloud APIs if local models unavailable
- **Unified Interface**: Same API for all LLM backends
- **Configuration Control**: Easy switching between local and cloud preferences

## Setup

### 1. Install Dependencies

```bash
pip install google-generativeai>=0.3.0
```

### 2. Get Google API Key

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Copy the key for use in TANGRAM

### 3. Configure Environment

```bash
export GOOGLE_API_KEY="your-api-key-here"
```

### 4. Update Configuration

Edit `config.py` to enable Gemini:

```python
LLM_CONFIG = {
    "provider": "unified",
    "prefer_local": False,  # Set to True to prefer local models
    "gemini": {
        "enabled": True,
        "model": "gemini-1.5-flash",  # or "gemini-1.5-pro"
        "timeout": 30
    }
}
```

## Usage

### Basic Usage

```python
from src.tangram.core.llm.local_llm_client import UnifiedLLMClient

# Initialize client
client = UnifiedLLMClient(
    prefer_local=False,
    gemini_model="gemini-1.5-flash"
)

# Generate response
response = client.generate_response(
    prompt="Analyze this scene",
    system_prompt="You are a robotics expert",
    max_tokens=1000
)

print(response['content'])
```

### Scene Analysis

```python
from src.tangram.core.llm.local_llm_client import UnifiedDeepSeekInterpreter

# Initialize interpreter
interpreter = UnifiedDeepSeekInterpreter(
    prefer_local=False,
    gemini_api_key="your-api-key"
)

# Analyze scene graph
result = interpreter.analyze_scene_graph(scene_graph_data)
print(result['scene_analysis'])
```

### Force Specific Backend

```python
# Force use of Gemini
response = client.generate_response(
    prompt="Your prompt",
    preferred_client="gemini"
)

# Force use of local model
response = client.generate_response(
    prompt="Your prompt", 
    preferred_client="local"
)
```

## Available Models

### Gemini Models

- **gemini-1.5-flash**: Fast, efficient model for most tasks
- **gemini-1.5-pro**: More capable model for complex reasoning

### Local Models

- **deepseek-r1:7b**: Local DeepSeek model via Ollama
- **deepseek-r1:latest**: Latest DeepSeek model

## Configuration Options

### Client Priority

```python
# Prefer local models, fallback to cloud
client = UnifiedLLMClient(prefer_local=True)

# Prefer cloud models, fallback to local
client = UnifiedLLMClient(prefer_local=False)
```

### Model Selection

```python
# Use specific Gemini model
client = UnifiedLLMClient(
    gemini_model="gemini-1.5-pro",
    prefer_local=False
)

# Use specific local model
client = UnifiedLLMClient(
    local_model="deepseek-r1:latest",
    prefer_local=True
)
```

## Error Handling

The unified client handles errors gracefully:

1. **Local Model Unavailable**: Automatically falls back to Gemini
2. **API Key Missing**: Clear error message with setup instructions
3. **Network Issues**: Retries and fallback to available backends
4. **Rate Limiting**: Proper error handling for API limits

## Performance Comparison

| Backend | Speed | Cost | Privacy | Capabilities |
|---------|-------|------|---------|--------------|
| Local (DeepSeek) | Medium | Free | High | Good |
| Gemini Flash | Fast | Low | Medium | Excellent |
| Gemini Pro | Medium | Medium | Medium | Superior |

## Testing

Run the example to test your integration:

```bash
export GOOGLE_API_KEY="your-api-key"
python examples/gemini_llm_example.py
```

## Security Considerations

- **API Keys**: Never commit API keys to version control
- **Environment Variables**: Use environment variables for sensitive data
- **Local Privacy**: Local models keep all data on-device
- **Cloud Privacy**: Gemini API follows Google's privacy policies

## Troubleshooting

### Common Issues

1. **"Google API key required"**
   - Set the `GOOGLE_API_KEY` environment variable
   - Ensure the API key is valid and active

2. **"Google Gemini SDK not available"**
   - Install the SDK: `pip install google-generativeai`

3. **"No LLM clients available"**
   - Ensure either local Ollama is running OR Gemini API key is set
   - Check network connectivity for cloud APIs

4. **"All LLM clients failed"**
   - Check local Ollama service status
   - Verify API key and network connectivity
   - Check API quotas and rate limits

### Debug Mode

Enable debug logging to troubleshoot issues:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Integration with TANGRAM Pipeline

The Gemini integration works seamlessly with existing TANGRAM components:

1. **Scene Analysis**: Enhanced scene understanding with Gemini's capabilities
2. **Task Planning**: Better robot task generation and planning
3. **Natural Language**: Improved natural language processing for GUI
4. **Fallback System**: Automatic switching when local models fail

## Cost Estimation

### Gemini API Pricing (approximate)

- **gemini-1.5-flash**: $0.075 per 1M input tokens, $0.30 per 1M output tokens
- **gemini-1.5-pro**: $1.25 per 1M input tokens, $5.00 per 1M output tokens

### Typical TANGRAM Usage

- Scene analysis: ~500 tokens per request
- Task planning: ~1000 tokens per request
- Estimated cost: $0.001-0.01 per scene analysis

## Best Practices

1. **Use Local First**: Set `prefer_local=True` for privacy and cost savings
2. **Choose Right Model**: Use `gemini-1.5-flash` for most tasks
3. **Handle Errors**: Always include error handling for API failures
4. **Monitor Usage**: Track API usage to manage costs
5. **Environment Variables**: Use environment variables for configuration

## Future Enhancements

- Support for additional cloud providers (OpenAI, Anthropic)
- Advanced prompt engineering and optimization
- Model performance benchmarking
- Cost monitoring and budgeting tools
- Multi-modal support (vision + text)