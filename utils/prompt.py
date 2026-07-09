"""
utils/prompt.py
Prompt construction shared by training and inference.
"""


def build_prompt(ex: dict, prompt: str) -> str:
    """
    Build the model input for training or inference:

        {prompt}\n\nContext:\n{ctx}\n\nQuestion:\n{qst}\n\n

    Args:
        ex:     dataset example with keys `question` and either
                `temporal_context` or `context`.
        prompt: instruction template text (see utils.io_gpu.load_prompt_for_lang).

    Returns:
        The formatted prompt string.
    """
    instr = prompt.strip()
    ctx = (ex.get("temporal_context", "") or ex.get("context", "") or "").strip()
    qst = (ex.get("question", "") or "").strip()

    return (
        f"{instr}\n\n"
        f"Context:\n{ctx}\n\n"
        f"Question:\n{qst}\n\n"
    )