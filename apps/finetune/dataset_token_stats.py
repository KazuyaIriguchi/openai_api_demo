import tiktoken
import numpy as np

# Token counting functions
encoding = tiktoken.get_encoding("cl100k_base")

# not exact!
# simplified from https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
def num_tokens_from_messages(messages, tokens_per_message=3, tokens_per_name=1):
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3
    return num_tokens

def num_assistant_tokens_from_messages(messages):
    num_tokens = 0
    for message in messages:
        if message["role"] == "assistant":
            num_tokens += len(encoding.encode(message["content"]))
    return num_tokens

def print_distribution(values, name):
    print(f"\n#### Distribution of {name}:")
    print(f"min / max: {min(values)}, {max(values)}")
    print(f"mean / median: {np.mean(values)}, {np.median(values)}")
    print(f"p5 / p95: {np.quantile(values, 0.1)}, {np.quantile(values, 0.9)}")

def dataset_token_stats(dataset):
    """
    Calculate token statistics for the dataset.

    Args:
    - dataset: List of conversations.

    Returns:
    - stats: Dictionary containing token related statistics.
    """
    n_missing_system = 0
    n_missing_user = 0
    n_messages = []
    convo_lens = []
    assistant_message_lens = []

    for ex in dataset:
        messages = ex["messages"]
        if not any(message["role"] == "system" for message in messages):
            n_missing_system += 1
        if not any(message["role"] == "user" for message in messages):
            n_missing_user += 1
        n_messages.append(len(messages))
        convo_lens.append(num_tokens_from_messages(messages))
        assistant_message_lens.append(num_assistant_tokens_from_messages(messages))

    print("Num examples missing system message:", n_missing_system)
    print("Num examples missing user message:", n_missing_user)
    print_distribution(n_messages, "num_messages_per_example")
    print_distribution(convo_lens, "num_total_tokens_per_example")
    print_distribution(assistant_message_lens, "num_assistant_tokens_per_example")
    n_too_long = sum(l > 4096 for l in convo_lens)
    print(f"\n{n_too_long} examples may be over the 4096 token limit, they will be truncated during fine-tuning")

    # Pricing and default n_epochs estimate
    MAX_TOKENS_PER_EXAMPLE = 4096

    MIN_TARGET_EXAMPLES = 100
    MAX_TARGET_EXAMPLES = 25000
    TARGET_EPOCHS = 3
    MIN_EPOCHS = 1
    MAX_EPOCHS = 25

    n_epochs = TARGET_EPOCHS
    n_train_examples = len(dataset)
    if n_train_examples * TARGET_EPOCHS < MIN_TARGET_EXAMPLES:
        n_epochs = min(MAX_EPOCHS, MIN_TARGET_EXAMPLES // n_train_examples)
    elif n_train_examples * TARGET_EPOCHS > MAX_TARGET_EXAMPLES:
        n_epochs = max(MIN_EPOCHS, MAX_TARGET_EXAMPLES // n_train_examples)

    n_billing_tokens_in_dataset = sum(min(MAX_TOKENS_PER_EXAMPLE, length) for length in convo_lens)
    print(f"Dataset has ~{n_billing_tokens_in_dataset} tokens that will be charged for during training")
    print(f"By default, you'll train for {n_epochs} epochs on this dataset")
    print(f"By default, you'll be charged for ~{n_epochs * n_billing_tokens_in_dataset} tokens")
    print("See pricing page to estimate total costs")

    stats = {
        "missing_system_messages": n_missing_system,
        "missing_user_messages": n_missing_user,
        "total_tokens": sum(convo_lens),
        "examples_over_limit": sum(l > 4096 for l in convo_lens),
        "distributions": {
            "num_messages_per_example": {
                "min": min(n_messages),
                "max": max(n_messages),
                "mean": np.mean(n_messages),
                "median": np.median(n_messages),
                "p5": np.quantile(n_messages, 0.1),
                "p95": np.quantile(n_messages, 0.9),
            },
            "num_total_tokens_per_example": {
                "min": min(convo_lens),
                "max": max(convo_lens),
                "mean": np.mean(convo_lens),
                "median": np.median(convo_lens),
                "p5": np.quantile(convo_lens, 0.1),
                "p95": np.quantile(convo_lens, 0.9),
            },
            "num_assistant_tokens_per_example": {
                "min": min(assistant_message_lens),
                "max": max(assistant_message_lens),
                "mean": np.mean(assistant_message_lens),
                "median": np.median(assistant_message_lens),
                "p5": np.quantile(assistant_message_lens, 0.1),
                "p95": np.quantile(assistant_message_lens, 0.9),
            },
        },
        "too_long": n_too_long,
        "has_tokens": n_billing_tokens_in_dataset,
        "n_epochs": n_epochs,
        "will_charge": n_epochs * n_billing_tokens_in_dataset,
    }

    return stats
