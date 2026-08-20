# Qwen3 ASR Refiner

<p align="center">
  <a href="https://huggingface.co/Aye10032"><img src="https://img.shields.io/badge/🤗_Hugging_Face-models-FFD21E" alt="Hugging Face models"></a>
  <a href="https://huggingface.co/datasets/Aye10032/WenetSpeech-Formal-Text"><img src="https://img.shields.io/badge/🤗_Dataset-WenetSpeech--Formal--Text-FFD21E" alt="Training dataset"></a>
  <img src="https://img.shields.io/badge/Python-3.13+-3776AB?logo=python&logoColor=white" alt="Python 3.13+">
  <img src="https://img.shields.io/badge/fine--tuning-LoRA-7C3AED" alt="LoRA fine-tuning">
  <img src="https://img.shields.io/badge/Transformers-5.0+-FF9D00" alt="Transformers 5.0+">
</p>

Qwen3 ASR Refiner 是基于 Qwen3 微调的中文 ASR 文本后处理模型：把包含语气词、重复表达和口语结构的识别文本整理为简洁、自然的书面语，同时尽量保持原意。

> 本仓库提供数据准备、LoRA 训练和权重导出代码，不存放模型权重。直接使用模型请前往下方的 Hugging Face 仓库。

## 模型与数据

| 资源 | Hugging Face |
| --- | --- |
| Qwen3-0.6B-ASR-Refiner | [`Aye10032/Qwen3-0.6B-ASR-Refiner`](https://huggingface.co/Aye10032/Qwen3-0.6B-ASR-Refiner) |
| Qwen3-1.7B-ASR-Refiner | [`Aye10032/Qwen3-1.7B-ASR-Refiner`](https://huggingface.co/Aye10032/Qwen3-1.7B-ASR-Refiner) |
| Qwen3-4B-ASR-Refiner | [`Aye10032/Qwen3-4B-ASR-Refiner`](https://huggingface.co/Aye10032/Qwen3-4B-ASR-Refiner) |
| 训练数据 | [`Aye10032/WenetSpeech-Formal-Text`](https://huggingface.co/datasets/Aye10032/WenetSpeech-Formal-Text) |

## Quickstart

以下示例使用 1.7B 版本；可替换为上表中的其他模型。

### Transformers

```bash
pip install -U torch transformers accelerate
```

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = 'Aye10032/Qwen3-1.7B-ASR-Refiner'
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype='auto',
    device_map='auto',
)

messages = [
    {
        'role': 'system',
        'content': '将中文口语转写改写为正式、自然的书面语。保持原意，不添加原文没有的信息，只输出改写后的文本。',
    },
    {'role': 'user', 'content': '呃这个事情吧我们之后再讨论一下。'},
]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    enable_thinking=False,
    return_dict=True,
    return_tensors='pt',
).to(model.device)
outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False)
result = tokenizer.decode(outputs[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
print(result)
```

### vLLM

```bash
pip install -U vllm openai
vllm serve Aye10032/Qwen3-1.7B-ASR-Refiner \
  --served-model-name qwen3-asr-refiner \
  --default-chat-template-kwargs '{"enable_thinking": false}'
```

服务启动后，可通过 OpenAI-compatible API 调用：

```python
from openai import OpenAI

client = OpenAI(base_url='http://localhost:8000/v1', api_key='EMPTY')
response = client.chat.completions.create(
    model='qwen3-asr-refiner',
    messages=[
        {
            'role': 'system',
            'content': '将中文口语转写改写为正式、自然的书面语。保持原意，不添加原文没有的信息，只输出改写后的文本。',
        },
        {'role': 'user', 'content': '呃这个事情吧我们之后再讨论一下。'},
    ],
    max_tokens=256,
    temperature=0,
)
print(response.choices[0].message.content)
```
