"""
utils/prompt.py
Prompt construction shared by training and inference.
"""


def build_prompt(ex: dict, prompt: str) -> str:
    """
    Build the prompt for training or inference.

    Both share the same base format:
        {instr}\n\nContext:\n{ctx}\n\nQuestion:\n{qst}\n\n

   
    Args:
        ex:            dataset example with keys:
                         prompt, temporal_context (or context), question
        for_inference: True for eval/inference, False for training

    Returns:
        Prompt string, ending with "<reasoning>" iff for_inference=True.
    """
    
    instr = prompt.strip()
    ctx   = (ex.get("temporal_context", "") or ex.get("context", "") or "").strip()
    qst   = (ex.get("question", "") or "").strip()

    base = (
        f"{instr}\n\n"
        f"Context:\n{ctx}\n\n"
        f"Question:\n{qst}\n\n"
    )

    
    return base