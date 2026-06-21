#!/bin/bash
# Build LAIT Granite Ollama model
echo "Building lait-granite model..."
ollama create lait-granite -f ollama/lait_granite/Modelfile
echo "Done! Model: lait-granite"
echo "Test: ollama run lait-granite"
