<div align="center">

# EVA V1

### Llama 3.1 8B | SFT + GRPO Reasoning Model

RECODED BY RERO

[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/rfrfrfrf2)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/R3RO)
[![Instagram](https://img.shields.io/badge/Instagram-E4405F?style=for-the-badge&logo=instagram&logoColor=white)](https://instagram.com/rero)
[![Telegram](https://img.shields.io/badge/Telegram-2CA5E0?style=for-the-badge&logo=telegram&logoColor=white)](https://t.me/rero)

</div>

---

## What is it?

EVA V1 is a reasoning model built on **Meta Llama 3.1 8B Instruct** using a two-phase training pipeline. The model is first fine-tuned with conversational data (SFT), then learns structured reasoning through reward functions (GRPO).

## Architecture

```
Llama 3.1 8B Instruct
        |
   LoRA (rank=32)
        |
   Phase 1: SFT ---- Conversational fine-tuning (FineTome-100k)
        |
   Phase 2: GRPO --- Reward-based reasoning training (GSM8K)
        |
     EVA V1
```

## Features

- **Two-phase training**: SFT + GRPO pipeline in a single notebook
- **Structured output**: Reasoning with `<reasoning>` and `<answer>` XML tags
- **Fast training**: 2x faster with 70% less VRAM using Unsloth
- **vLLM integration**: Fast inference with `fast_inference=True`
- **Memory management**: Automatic memory cleanup between phases
- **Flexible export**: LoRA, merged 16bit/4bit, GGUF (q8_0, q4_k_m, q5_k_m, f16)

## Requirements

- Google Colab (free Tesla T4) or CUDA-capable GPU
- Python 3.10+
- Minimum 15GB VRAM (with 4-bit quantization)

## Installation

```bash
pip install unsloth vllm
pip install transformers==4.56.2
pip install --no-deps trl==0.22.2
```

## Training Pipeline

### Phase 1: Supervised Fine-Tuning (SFT)

The model is fine-tuned with conversational data:

- **Dataset**: [mlabonne/FineTome-100k](https://huggingface.co/datasets/mlabonne/FineTome-100k) (ShareGPT format)
- **Custom Chat Template**: SYSTEM_MESSAGE is automatically embedded into every conversation
- **System role filter**: Prevents duplicate system messages
- **Training**: SFTTrainer, 1 epoch, lr=2e-5, adamw_8bit

| Parameter | Value |
|---|---|
| Batch size | 2 |
| Gradient accumulation | 4 |
| Epochs | 1 |
| Warmup steps | 20 |
| Learning rate | 2e-5 |
| Optimizer | adamw_8bit |
| Scheduler | linear |
| Max seq length | 2048 |

### Phase 2: Group Relative Policy Optimization (GRPO)

After SFT, the model learns structured reasoning through reward functions:

- **Dataset**: [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) (math problems)
- **Output format**: `<reasoning>...</reasoning><answer>...</answer>`
- **Training**: GRPOTrainer, 1 epoch, lr=2e-5, paged_adamw_8bit

| Parameter | Value |
|---|---|
| Batch size | 1 |
| Num generations | 6 |
| Epochs | 1 |
| Warmup steps | 20 |
| Learning rate | 2e-5 |
| Optimizer | paged_adamw_8bit |
| Scheduler | cosine |
| Max prompt length | 256 |
| Max completion length | 1792 |

### Reward Functions

The model is trained with 5 different reward signals:

| Function | Metric | Max Reward |
|---|---|---|
| `correctness_reward_func` | Is the answer correct? | 2.0 |
| `int_reward_func` | Is the answer numeric? | 0.5 |
| `strict_format_reward_func` | Exact XML format match? | 0.5 |
| `soft_format_reward_func` | Loose XML format match? | 0.5 |
| `xmlcount_reward_func` | Partial XML tag scoring | 0.5 |

## Model Configuration

### Base Model
- **Model**: `unsloth/meta-Llama-3.1-8B-Instruct`
- **Quantization**: 4-bit (bitsandbytes)
- **Max sequence length**: 2048

### LoRA
- **Rank**: 32
- **Alpha**: 32
- **Target modules**: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- **Gradient checkpointing**: unsloth (long context support)

### System Prompt

```
You are a helpful reasoning assistant. You think step by step
and provide clear, structured answers. Use <reasoning> tags for
your thought process and <answer> tags for your final answer.
```

### GRPO Reasoning Format

```xml
<reasoning>
Step-by-step thought process...
</reasoning>
<answer>
Final answer
</answer>
```

## Usage

### Running on Colab

1. Open the notebook in Google Colab
2. Select Runtime > Run all
3. Phase 1 (SFT) takes ~15-20 minutes
4. Phase 2 (GRPO) takes ~30-45 minutes

### Inference

```python
from vllm import SamplingParams

text = tokenizer.apply_chat_template([
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "Your question here"},
], tokenize=False, add_generation_prompt=True)

sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=1024,
)

output = model.fast_generate(
    [text],
    sampling_params=sampling_params,
    lora_request=model.load_lora("grpo_saved_lora"),
)[0].outputs[0].text
```

### Example Output

```xml
<reasoning>
To calculate pi we can use the Leibniz formula:
pi/4 = 1 - 1/3 + 1/5 - 1/7 + ...

Let's compute the first 1000 terms:
Sum = 1 - 0.3333 + 0.2 - 0.1429 + ...
Sum = 0.7854

pi = 4 * 0.7854 = 3.1416
</reasoning>
<answer>
3.1416
</answer>
```

## Export Options

### LoRA Adapters
```python
model.save_lora("grpo_saved_lora")
```

### Merged Model (16-bit)
```python
model.save_pretrained_merged("model_16bit", tokenizer, save_method="merged_16bit")
```

### GGUF (llama.cpp / Ollama)
```python
model.save_pretrained_gguf("model", tokenizer, quantization_method="q4_k_m")
```

| Format | Size | Use Case |
|---|---|---|
| LoRA | ~100MB | Lightweight, loaded as adapter |
| Merged 16bit | ~16GB | VLLM deployment |
| Merged 4bit | ~4GB | Low memory deployment |
| GGUF q8_0 | ~8GB | llama.cpp, Ollama |
| GGUF q4_k_m | ~4GB | llama.cpp, Ollama (recommended) |
| GGUF q5_k_m | ~5GB | llama.cpp, Ollama |

## File Structure

```
EVA_V1.ipynb
|
|-- Setup (Cell 4-5)
|-- Model Setup (Cell 8)
|-- Phase 1: SFT
|   |-- System Message (Cell 10)
|   |-- Chat Template (Cell 11)
|   |-- Dataset Loading (Cell 12)
|   |-- SFT Trainer (Cell 13)
|   |-- SFT Training (Cell 14)
|   |-- Template Reset (Cell 15)
|
|-- Phase 2: GRPO
|   |-- Data Prep + Rewards (Cell 18)
|   |-- GRPO Config (Cell 20)
|   |-- GRPO Training (Cell 22)
|
|-- Inference (Cell 24, 28)
|-- Save/Export (Cell 31, 33)
```

## Customization

### Custom System Message
Change the `SYSTEM_MESSAGE` content in Cell 10.

### Custom Dataset
Replace `load_dataset("mlabonne/FineTome-100k")` in Cell 12 with your own dataset. ShareGPT format is required.

### Longer Training
- SFT: Increase `num_train_epochs=1` in Cell 13 (2-3 epochs recommended)
- GRPO: Increase `num_train_epochs=1` in Cell 20 (2-3 epochs recommended)

### Different Base Model
Change the `model_name` parameter in Cell 8. Unsloth-supported models:
- `unsloth/meta-Llama-3.1-8B-Instruct`
- `unsloth/Meta-Llama-3.1-70B-bnb-4bit`
- `unsloth/Llama-3.2-3B-Instruct-bnb-4bit`
- `unsloth/mistral-7b-instruct-v0.3-bnb-4bit`
- `unsloth/gemma-2-9b-bnb-4bit`

## Tech Stack

| Component | Version |
|---|---|
| Base Model | Llama 3.1 8B Instruct |
| Framework | Unsloth |
| Training | trl 0.22.2 (SFTTrainer + GRPOTrainer) |
| Inference | vLLM |
| Transformers | 4.56.2 |
| Quantization | bitsandbytes (4-bit) |
| LoRA | PEFT (rank 32) |

## License

This notebook is licensed under [LGPL-3.0](https://github.com/unslothai/notebooks?tab=LGPL-3.0-1-ov-file#readme).

---

<div align="center">

### RECODED BY RERO

[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/rfrfrfrf2)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/R3RO)
[![Instagram](https://img.shields.io/badge/Instagram-E4405F?style=for-the-badge&logo=instagram&logoColor=white)](https://instagram.com/rero)
[![Telegram](https://img.shields.io/badge/Telegram-2CA5E0?style=for-the-badge&logo=telegram&logoColor=white)](https://t.me/rero)

</div>
