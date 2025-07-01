# 🌩️ DeepSeek on Thunder Compute Setup Guide

This guide shows you how to run DeepSeek locally on Thunder Compute instead of using the external API.

## 🎯 Why Local DeepSeek?

- **Data Privacy**: Keep your scene data on your Thunder instance
- **No API Costs**: Only pay for Thunder Compute time (~$0.50/run instead of $0.50 + API fees)
- **Faster Inference**: Reduce network latency
- **Offline Capability**: Works without internet after setup

## 💰 Updated Cost Estimate

### **With Local DeepSeek:**
- **A100 Instance**: ~$1.70/hour (need GPU for LLM inference)
- **15-minute simulation**: ~$0.43
- **API costs**: $0 (no external calls)
- **Total**: ~$0.43 per run

### **Recommended Configuration:**
```bash
# A100 needed for local LLM inference
tnr create --gpu a100 --vcpus 8 --mode prototyping
```

## 🚀 Setup Instructions

### **Step 1: Create Thunder Instance**
```bash
# Create A100 instance (required for local LLM)
tnr create --gpu a100 --vcpus 8 --mode prototyping

# Get instance status
tnr status
```

### **Step 2: Upload Setup Script**
```bash
# Copy setup script to Thunder instance
tnr scp docs/scripts/setup_deepseek_thunder.sh 0:/tmp/

# Make executable
tnr connect 0 "chmod +x /tmp/setup_deepseek_thunder.sh"
```

### **Step 3: Run Setup on Thunder Instance**
```bash
# Connect and run setup (takes 10-20 minutes)
tnr connect 0 "bash /tmp/setup_deepseek_thunder.sh"
```

**The setup script will:**
- Install Ollama
- Download DeepSeek-R1 model (~15GB)
- Start local inference server
- Test the installation
- Create service scripts

### **Step 4: Upload TANGRAM Code**
```bash
# Upload entire TANGRAM project
tnr scp -r . 0:/tmp/tangram/

# Or just upload specific files if project is large
tnr scp src/ config.py main.py 0:/tmp/tangram/
```

### **Step 5: Test Local LLM**
```bash
# Connect to instance and test
tnr connect 0

# On the Thunder instance:
cd /tmp/tangram
python tests/test_local_llm.py
```

**Expected output:**
```
🎉 All tests passed! Local LLM integration is ready.
```

### **Step 6: Run TANGRAM with Local DeepSeek**
```bash
# On Thunder instance:
python main.py --input video.mp4 --mode full

# Or run demo with local LLM:
python examples/demo.py --use-mock-data
```

## ⚙️ Configuration Details

The system automatically detects and uses local LLM when configured:

```python
# In config.py - already set up for you:
LLM_CONFIG = {
    "provider": "local",  # Uses local Ollama
    "local": {
        "enabled": True,
        "host": "localhost", 
        "port": 11434,
        "model": "deepseek-r1:latest",
        "fallback_to_api": True  # Falls back if local fails
    }
}
```

## 🔧 Troubleshooting

### **Ollama Not Starting:**
```bash
# Check if running
curl http://localhost:11434/api/tags

# Restart service
bash /tmp/start_ollama.sh

# Check logs
tail -f /tmp/ollama.log
```

### **Model Not Found:**
```bash
# List available models
ollama list

# Pull DeepSeek model manually
ollama pull deepseek-r1:latest

# Or try smaller model
ollama pull deepseek-coder:6.7b
```

### **TANGRAM Import Errors:**
```bash
# Install Python dependencies
pip3 install --user requests ollama-python

# Check TANGRAM imports
python -c "from src.tangram.core.llm.local_llm_client import LocalLLMClient; print('OK')"
```

## 🔄 Switching Between Local and API

You can easily switch between local and API-based inference:

### **Use Local DeepSeek:**
```python
# In config.py
LLM_CONFIG["provider"] = "local"
```

### **Use API DeepSeek:**
```python
# In config.py  
LLM_CONFIG["provider"] = "deepseek"
```

### **Automatic Fallback:**
The system automatically falls back to API if local inference fails.

## 📊 Performance Comparison

| Mode | Instance | Cost/15min | Inference Speed | Setup Time |
|------|----------|------------|----------------|------------|
| **API** | T4 | $0.15 + API | Fast | 0 min |
| **Local** | A100 | $0.43 | Very Fast | 15 min |

## 🎉 Ready to Go!

After setup, your TANGRAM pipeline will:
1. ✅ Run object detection/tracking on Thunder GPU
2. ✅ Execute PyBullet simulation locally  
3. ✅ Perform LLM scene analysis locally (no API calls)
4. ✅ Keep all data private on your Thunder instance

**Total cost: ~$0.43 per simulation with complete data privacy!**