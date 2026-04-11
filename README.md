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

## Nedir?

EVA V1, **Meta Llama 3.1 8B Instruct** temelinde iki asamali egitim pipeline'i ile olusturulmus bir reasoning (muhakeme) modelidir. Model once konusma verileriyle fine-tune edilir (SFT), ardindan odul fonksiyonlariyla yapilandirilmis muhakeme ogrenir (GRPO).

## Mimari

```
Llama 3.1 8B Instruct
        |
   LoRA (rank=32)
        |
   Phase 1: SFT ---- Konusma verisiyle fine-tune (FineTome-100k)
        |
   Phase 2: GRPO --- Odul fonksiyonlariyla reasoning egitimi (GSM8K)
        |
     EVA V1
```

## Ozellikler

- **Iki asamali egitim**: SFT + GRPO pipeline'i tek notebook'ta
- **Yapilandirilmis cikti**: `<reasoning>` ve `<answer>` XML tag'leri ile muhakeme
- **Hizli egitim**: Unsloth ile 2x hizli, %70 daha az VRAM
- **vLLM entegrasyonu**: `fast_inference=True` ile hizli cikti uretimi
- **Bellek yonetimi**: Fazlar arasi otomatik bellek temizligi
- **Esnek kaydetme**: LoRA, merged 16bit/4bit, GGUF (q8_0, q4_k_m, q5_k_m, f16)

## Gereksinimler

- Google Colab (Tesla T4 ucretsiz) veya CUDA destekli GPU
- Python 3.10+
- Minimum 15GB VRAM (4-bit quantization ile)

## Kurulum

```bash
pip install unsloth vllm
pip install transformers==4.56.2
pip install --no-deps trl==0.22.2
```

## Egitim Pipeline'i

### Phase 1: Supervised Fine-Tuning (SFT)

Model, konusma verileriyle fine-tune edilir. Bu asamada:

- **Dataset**: [mlabonne/FineTome-100k](https://huggingface.co/datasets/mlabonne/FineTome-100k) (ShareGPT formati)
- **Custom Chat Template**: SYSTEM_MESSAGE otomatik olarak her konusmaya gomulur
- **System role filtresi**: Cift system message onlenir
- **Egitim**: SFTTrainer, 200 step, lr=2e-4, adamw_8bit

| Parametre | Deger |
|---|---|
| Batch size | 2 |
| Gradient accumulation | 4 |
| Max steps | 200 |
| Learning rate | 2e-4 |
| Optimizer | adamw_8bit |
| Scheduler | linear |
| Max seq length | 2048 |

### Phase 2: Group Relative Policy Optimization (GRPO)

SFT sonrasi, model odul fonksiyonlariyla yapilandirilmis muhakeme ogrenir:

- **Dataset**: [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) (matematik problemleri)
- **Cikti formati**: `<reasoning>...</reasoning><answer>...</answer>`
- **Egitim**: GRPOTrainer, 250 step, lr=5e-6, paged_adamw_8bit

| Parametre | Deger |
|---|---|
| Batch size | 1 |
| Num generations | 6 |
| Max steps | 250 |
| Learning rate | 5e-6 |
| Optimizer | paged_adamw_8bit |
| Scheduler | cosine |
| Max prompt length | 256 |
| Max completion length | 1792 |

### Odul Fonksiyonlari

Model 5 farkli odul sinyali ile egitilir:

| Fonksiyon | Olcum | Max Odul |
|---|---|---|
| `correctness_reward_func` | Dogru cevap mi? | 2.0 |
| `int_reward_func` | Cevap sayisal mi? | 0.5 |
| `strict_format_reward_func` | XML formati tam dogru mu? | 0.5 |
| `soft_format_reward_func` | XML formati gevsek uyuyor mu? | 0.5 |
| `xmlcount_reward_func` | XML tag'leri kismi puan | 0.5 |

## Model Yapilandirmasi

### Base Model
- **Model**: `unsloth/meta-Llama-3.1-8B-Instruct`
- **Quantization**: 4-bit (bitsandbytes)
- **Max sequence length**: 2048

### LoRA
- **Rank**: 32
- **Alpha**: 32
- **Target modules**: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- **Gradient checkpointing**: unsloth (uzun bağlam destegi)

### System Prompt

```
You are a helpful reasoning assistant. You think step by step
and provide clear, structured answers. Use <reasoning> tags for
your thought process and <answer> tags for your final answer.
```

### GRPO Reasoning Formati

```xml
<reasoning>
Adim adim dusunce sureci...
</reasoning>
<answer>
Nihai cevap
</answer>
```

## Kullanim

### Colab'da Calistirma

1. Notebook'u Google Colab'da acin
2. Runtime > Run all secin
3. Phase 1 (SFT) ~15-20 dakika surer
4. Phase 2 (GRPO) ~30-45 dakika surer

### Inference

```python
from vllm import SamplingParams

text = tokenizer.apply_chat_template([
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "Sorunuz buraya"},
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

### Ornek Cikti

```xml
<reasoning>
Pi sayisini hesaplamak icin Leibniz formulu kullanabiliriz:
pi/4 = 1 - 1/3 + 1/5 - 1/7 + ...

Ilk 1000 terimi hesaplayalim:
Toplam = 1 - 0.3333 + 0.2 - 0.1429 + ...
Toplam ≈ 0.7854

pi ≈ 4 * 0.7854 = 3.1416
</reasoning>
<answer>
3.1416
</answer>
```

## Kaydetme Secenekleri

### LoRA Adaptorleri
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

| Format | Boyut | Kullanim |
|---|---|---|
| LoRA | ~100MB | Hafif, adapter olarak yuklenir |
| Merged 16bit | ~16GB | VLLM deployment |
| Merged 4bit | ~4GB | Dusuk bellek deployment |
| GGUF q8_0 | ~8GB | llama.cpp, Ollama |
| GGUF q4_k_m | ~4GB | llama.cpp, Ollama (onerilir) |
| GGUF q5_k_m | ~5GB | llama.cpp, Ollama |

## Dosya Yapisi

```
EVA_V1.ipynb
|
|-- Kurulum (Cell 4-5)
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

## Ozellestirme

### Farkli System Message
Cell 10'daki `SYSTEM_MESSAGE` iceriğini degistirin.

### Farkli Dataset
Cell 12'deki `load_dataset("mlabonne/FineTome-100k")` satirini kendi dataset'inizle degistirin. ShareGPT formati gereklidir.

### Daha Uzun Egitim
- SFT: Cell 13'te `max_steps=200` degerini artirin (500-1000 onerilir)
- GRPO: Cell 20'de `max_steps=250` degerini artirin (500-1000 onerilir)

### Farkli Base Model
Cell 8'deki `model_name` parametresini degistirin. Unsloth destekli modeller:
- `unsloth/meta-Llama-3.1-8B-Instruct`
- `unsloth/Meta-Llama-3.1-70B-bnb-4bit`
- `unsloth/Llama-3.2-3B-Instruct-bnb-4bit`
- `unsloth/mistral-7b-instruct-v0.3-bnb-4bit`
- `unsloth/gemma-2-9b-bnb-4bit`

## Teknoloji Stack

| Bilesen | Versiyon |
|---|---|
| Base Model | Llama 3.1 8B Instruct |
| Framework | Unsloth |
| Training | trl 0.22.2 (SFTTrainer + GRPOTrainer) |
| Inference | vLLM |
| Transformers | 4.56.2 |
| Quantization | bitsandbytes (4-bit) |
| LoRA | PEFT (rank 32) |

## Lisans

Bu notebook [LGPL-3.0](https://github.com/unslothai/notebooks?tab=LGPL-3.0-1-ov-file#readme) lisansi altindadir.

---

<div align="center">

### RECODED BY RERO

[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/rfrfrfrf2)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/R3RO)
[![Instagram](https://img.shields.io/badge/Instagram-E4405F?style=for-the-badge&logo=instagram&logoColor=white)](https://instagram.com/rero)
[![Telegram](https://img.shields.io/badge/Telegram-2CA5E0?style=for-the-badge&logo=telegram&logoColor=white)](https://t.me/rero)

</div>
