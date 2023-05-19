from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained(
    "D:\Projects\desistdaydream\python-transformers\chatglm-6b-int4",
    trust_remote_code=True,
)
model = AutoModel.from_pretrained(
    "D:\Projects\DesistDaydream\python-transformers\chatglm-6b-int4",
    trust_remote_code=True,
).float()
model = model.eval()
response, history = model.chat(tokenizer, "你好", history=[])
print(response)
