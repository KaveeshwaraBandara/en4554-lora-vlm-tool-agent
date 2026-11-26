# Part 2 – LoRA-finetuning a small LLM for tool calling

**Model:** `Qwen/Qwen2.5-0.5B-Instruct`. Its chat template accepts `tools=` and emits Hermes-style `<tool_call>{json}</tool_call>`.

## Plan

1. **Tools (4):** calculator, unit converter, fake weather lookup (made-up cities), fake price lookup. Made-up data forces the model to call the tool instead of answering from memory.
2. **Agent loop:** generate → parse `<tool_call>` → execute → feed the `tool` message back → stop on a plain-text answer or after 4 steps.
3. **Data:** template-generated. About 1,200 train and 100 test; about 25% need no tool, and a few need two calls. The test set uses held-out phrasings and held-out values.
4. **Baseline vs LoRA:** same test set. LoRA r=16 on all linear layers, loss only on assistant turns.
5. **Metrics:** call validity, call accuracy, abstention, task success. Also a few-shot prompt-only baseline, so the "would a better prompt be enough?" question can be answered with data.
6. **Failure analysis:** at least 10 failures per model, grouped into categories (fixed / not fixed / introduced by LoRA).
