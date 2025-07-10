# 🔒 LOCAL-ONLY DeepSeek on Thunder Compute

**ZERO EXTERNAL API CALLS - 100% LOCAL INFERENCE**

This guide sets up DeepSeek to run entirely on your Thunder Compute instance with **NO external API calls whatsoever**.

## 🎯 Why LOCAL-ONLY DeepSeek?

- **🔒 Complete Data Privacy**: Zero external API calls - all data stays on your instance
- **💰 No API Costs**: Eliminate all DeepSeek API fees (was ~$0.005/call)
- **🚀 Faster Inference**: No network latency to external services  
- **🌐 Offline Capable**: Works without internet after setup
- **🛡️ Security Compliance**: No data leaves your Thunder instance

## 💰 Cost Analysis (LOCAL-ONLY)

### **Complete Local Setup:**
- **A100 Instance**: ~$1.70/hour (required for local LLM inference)
- **15-minute simulation**: ~$0.43
- **External API costs**: **$0** (ZERO external calls)
- **Data egress costs**: **$0** (no data leaves Thunder instance)
- **Total**: ~$0.43 per run with **complete privacy**

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

## ⚙️ Configuration Details (LOCAL-ONLY)

**NO EXTERNAL API FALLBACKS** - System enforces local-only operation:

```python
# In config.py - ZERO external API calls:
LLM_CONFIG = {
    "provider": "local",  # MUST be local
    "local": {
        "enabled": True,
        "host": "localhost", 
        "port": 11434,
        "model": "deepseek-r1:latest",
        "fallback_to_api": False,  # DISABLED - no external APIs
        "require_local": True      # Fail if local unavailable
    }
}
```

**Enforcement:** System will **crash with error** if local LLM unavailable rather than making external API calls.

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

## 🔒 LOCAL-ONLY Operation

**NO API OPTIONS** - System is configured for local-only operation:

### **Enforced Local Operation:**
```python
# ONLY valid configuration:
LLM_CONFIG["provider"] = "local"  # REQUIRED
LLM_CONFIG["local"]["fallback_to_api"] = False  # DISABLED
```

### **Error Behavior:**
- **If local LLM unavailable**: System crashes with clear error message
- **No silent fallbacks**: Never makes external API calls
- **Explicit failures**: You know immediately if local setup has issues

### **Privacy Guarantee:**
✅ **ZERO network calls to external LLM APIs**  
✅ **All inference happens on your Thunder instance**  
✅ **Complete data isolation**

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