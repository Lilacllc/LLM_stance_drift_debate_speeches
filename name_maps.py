"""
Shared model name mappings for LaTeX tables and plot labels.
"""

# LaTeX-formatted names (used in .tex table output)
MODEL_NAME_LATEX = {
    "gpt_4o_mini": r"\texttt{gpt-4o-mini}",
    "gpt_4_1": r"\texttt{gpt-4.1}",
    "gpt_3_5_turbo": r"\texttt{gpt-3.5-turbo}",
    "gemma_3n_e4b": r"\texttt{gemma-3n-E4B}",
    "llama3_3_70b": r"\texttt{Llama-3.3-70B}",
    "llama3_1_8b": r"\texttt{Llama-3.1-8B}",
    "llama4_maverick": r"\texttt{Llama-4-Maverick}",
    "qwen3_a3b": r"\texttt{Qwen3-A3B}",
    "gpt_4o_mini_reversed": r"\texttt{gpt-4o-mini (reversed)}",
    "gpt_4o_mini_temp0": r"\texttt{gpt-4o-mini (temp0)}",
    "gpt_4o_mini_multiple_summarization": r"\texttt{gpt-4o-mini (multiple extraction)}",
    "gpt_4o_mini_in_context": r"\texttt{gpt-4o-mini (in-context)}",
    "gpt_4o_mini_assert": r"\texttt{gpt-4o-mini (assert)}",
}

# Plot-formatted names (used in matplotlib xtick labels, with \n for line breaks)
MODEL_NAME_PLOT = {
    "gpt_4o_mini": "gpt-4o-mini",
    "gpt_4_1": "gpt-4.1",
    "gpt_3_5_turbo": "gpt-3.5-turbo",
    "gemma_3n_e4b": "gemma-3n-\nE4B",
    "llama3_3_70b": "Llama-3.3-70B",
    "llama3_1_8b": "Llama-3.1-8B",
    "llama4_maverick": "Llama-4-\nMaverick",
    "qwen3_a3b": "Qwen3-A3B",
    "gpt_4o_mini_reversed": "gpt-4o-mini\n(reversed)",
    "gpt_4o_mini_temp0": "gpt-4o-mini\n(temp0)",
    "gpt_4o_mini_multiple_summarization": "gpt-4o-mini\n(multiple\nextraction)",
    "gpt_4o_mini_in_context": "gpt-4o-mini\n(in-context)",
    "gpt_4o_mini_assert": "gpt-4o-mini\n(assert)",
}


def get_latex_name(model_key):
    """Get LaTeX-formatted model name, with fallback to escaped raw name."""
    return MODEL_NAME_LATEX.get(model_key, model_key.replace("_", r"\_"))


def get_plot_name(model_key):
    """Get plot-formatted model name, with fallback to raw name."""
    return MODEL_NAME_PLOT.get(model_key, model_key)
