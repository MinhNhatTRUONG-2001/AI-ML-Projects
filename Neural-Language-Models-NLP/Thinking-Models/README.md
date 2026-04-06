# Thinking models
This is the big & final assignment of the course. The main workflow is a dataset of questions, their chain-of-thought (COT) texts and final answers are randomly generated and validated using external AI models (Llama from Groq, gpt-oss from Ollama and/or OpenRouter) with Camel AI; then that dataset is used with Unsloth for fine-tuning a small DeepSeek R1 model (LoRA is also applied in this process).

After cloning the project, please change the file name `env-template.txt` to `.env`, then follow the instruction in the `NhatTruong_ThinkingModelsInstruction.pdf` file.

- Camel AI: https://www.camel-ai.org/
- Unsloth: https://unsloth.ai/
