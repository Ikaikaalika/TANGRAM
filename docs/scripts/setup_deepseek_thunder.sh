#!/bin/bash
"""
Thunder Compute DeepSeek Setup Script

This script sets up DeepSeek model hosting on a Thunder Compute instance
using Ollama for local LLM inference.

Usage:
    # On Thunder Compute instance:
    bash setup_deepseek_thunder.sh

    # Or run remotely:
    tnr scp setup_deepseek_thunder.sh 0:/tmp/
    tnr connect 0 "bash /tmp/setup_deepseek_thunder.sh"

Author: TANGRAM Team
License: MIT
"""

set -e  # Exit on any error

echo "🌩️ Setting up DeepSeek on Thunder Compute"
echo "=========================================="

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Log function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

error() {
    echo -e "${RED}[ERROR] $1${NC}"
    exit 1
}

# Check if running on Thunder Compute
check_thunder_environment() {
    log "Checking Thunder Compute environment..."
    
    if [ "$USER" != "ubuntu" ]; then
        warn "Not running as ubuntu user. This script is designed for Thunder Compute instances."
    fi
    
    # Check for GPU
    if command -v nvidia-smi &> /dev/null; then
        log "GPU detected: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)"
    else
        warn "No GPU detected. DeepSeek will run on CPU (slower)."
    fi
}

# Install Ollama
install_ollama() {
    log "Installing Ollama..."
    
    if command -v ollama &> /dev/null; then
        log "Ollama already installed: $(ollama --version)"
        return
    fi
    
    # Download and install Ollama
    curl -fsSL https://ollama.ai/install.sh | sh
    
    if [ $? -eq 0 ]; then
        log "Ollama installed successfully"
    else
        error "Failed to install Ollama"
    fi
}

# Start Ollama service
start_ollama_service() {
    log "Starting Ollama service..."
    
    # Start Ollama in background
    nohup ollama serve > /tmp/ollama.log 2>&1 &
    OLLAMA_PID=$!
    
    # Wait for service to be ready
    for i in {1..30}; do
        if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
            log "Ollama service started successfully (PID: $OLLAMA_PID)"
            return
        fi
        echo "Waiting for Ollama service... ($i/30)"
        sleep 2
    done
    
    error "Ollama service failed to start"
}

# Pull DeepSeek model
pull_deepseek_model() {
    log "Pulling DeepSeek model..."
    
    # Check available models
    log "Available DeepSeek models:"
    ollama list | grep -i deepseek || log "No DeepSeek models currently installed"
    
    # Try to pull deepseek-r1 (reasoning model)
    MODEL_NAME="deepseek-r1:latest"
    log "Pulling $MODEL_NAME (this may take 10-20 minutes)..."
    
    if ollama pull $MODEL_NAME; then
        log "Successfully pulled $MODEL_NAME"
    else
        warn "Failed to pull $MODEL_NAME, trying alternative..."
        
        # Try smaller model as fallback
        MODEL_NAME="deepseek-coder:6.7b"
        log "Pulling fallback model $MODEL_NAME..."
        
        if ollama pull $MODEL_NAME; then
            log "Successfully pulled fallback model $MODEL_NAME"
        else
            error "Failed to pull any DeepSeek model"
        fi
    fi
    
    # Verify model is available
    log "Verifying model installation..."
    if ollama list | grep -q "deepseek"; then
        log "DeepSeek model(s) installed:"
        ollama list | grep deepseek
    else
        error "No DeepSeek models found after installation"
    fi
}

# Test local inference
test_inference() {
    log "Testing local inference..."
    
    # Get the installed model name
    MODEL_NAME=$(ollama list | grep deepseek | head -1 | awk '{print $1}')
    
    if [ -z "$MODEL_NAME" ]; then
        error "No DeepSeek model found for testing"
    fi
    
    log "Testing with model: $MODEL_NAME"
    
    # Test prompt
    TEST_PROMPT="Analyze this scene: A red apple is on a table next to a blue cup. Generate a simple robot task."
    
    # Make test request
    RESPONSE=$(curl -s -X POST http://localhost:11434/api/generate \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"$MODEL_NAME\",
            \"prompt\": \"$TEST_PROMPT\",
            \"stream\": false,
            \"options\": {\"temperature\": 0.7, \"num_predict\": 100}
        }")
    
    if [ $? -eq 0 ] && [ -n "$RESPONSE" ]; then
        log "✅ Local inference test successful!"
        echo "Response preview: $(echo $RESPONSE | jq -r '.response' | head -c 100)..."
    else
        error "Local inference test failed"
    fi
}

# Configure for TANGRAM integration
setup_tangram_integration() {
    log "Setting up TANGRAM integration..."
    
    # Create configuration file
    cat > /tmp/tangram_llm_config.json << EOF
{
    "local_llm": {
        "enabled": true,
        "provider": "ollama",
        "host": "localhost",
        "port": 11434,
        "model": "$(ollama list | grep deepseek | head -1 | awk '{print $1}')",
        "fallback_to_api": true
    },
    "setup_completed": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "thunder_instance": "$(hostname)"
}
EOF
    
    log "Configuration saved to /tmp/tangram_llm_config.json"
    cat /tmp/tangram_llm_config.json
}

# Install Python dependencies
install_python_deps() {
    log "Installing Python dependencies..."
    
    # Update package list
    sudo apt-get update -qq
    
    # Install pip if not present
    if ! command -v pip3 &> /dev/null; then
        sudo apt-get install -y python3-pip
    fi
    
    # Install required packages
    pip3 install --user requests ollama-python
    
    log "Python dependencies installed"
}

# Create service script
create_service_script() {
    log "Creating Ollama service script..."
    
    cat > /tmp/start_ollama.sh << 'EOF'
#!/bin/bash
# TANGRAM Ollama Service Startup Script

echo "Starting Ollama for TANGRAM..."
export OLLAMA_HOST=0.0.0.0:11434
export OLLAMA_MODELS_DIR=/tmp/ollama_models

# Start Ollama
nohup ollama serve > /tmp/ollama.log 2>&1 &
echo $! > /tmp/ollama.pid

echo "Ollama started. PID: $(cat /tmp/ollama.pid)"
echo "Logs: tail -f /tmp/ollama.log"
echo "Stop: kill $(cat /tmp/ollama.pid)"
EOF
    
    chmod +x /tmp/start_ollama.sh
    log "Service script created at /tmp/start_ollama.sh"
}

# Main setup function
main() {
    log "Starting DeepSeek setup on Thunder Compute..."
    
    check_thunder_environment
    install_python_deps
    install_ollama
    start_ollama_service
    pull_deepseek_model
    test_inference
    setup_tangram_integration
    create_service_script
    
    echo ""
    log "🎉 DeepSeek setup completed successfully!"
    echo ""
    echo "Next steps:"
    echo "1. In your TANGRAM config, set LLM provider to 'local'"
    echo "2. Run: python -c \"from src.tangram.core.llm.local_llm_client import LocalLLMClient; client = LocalLLMClient(); print('Local LLM ready!')\""
    echo "3. If Ollama stops, restart with: bash /tmp/start_ollama.sh"
    echo ""
    echo "Configuration details:"
    cat /tmp/tangram_llm_config.json
}

# Run main function
main "$@"