from __future__ import annotations


def build_prompts(n_train: int = 1200, n_eval: int = 256) -> tuple[list[str], list[str]]:
    train = [
        f"Research note {i}: predict the next token after this sentence fragment about interfaces and KL divergence"
        for i in range(n_train)
    ]
    eval_ = [
        f"Evaluation prompt {i}: one-token next-token prediction for extraction robustness"
        for i in range(n_eval)
    ]
    return train, eval_
