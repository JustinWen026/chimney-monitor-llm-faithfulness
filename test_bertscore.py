from bert_score import score

candidate = ["你的模型生成報告文字"]
reference = ["標準答案報告文字"]

P, R, F1 = score(
    candidate,
    reference,
    lang="zh",
    model_type="hfl/chinese-roberta-wwm-ext",
    num_layers=12,
    device="cuda",
)
print(f"F1: {F1.item():.4f}")