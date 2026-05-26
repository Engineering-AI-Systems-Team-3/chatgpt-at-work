SYSTEM_PROMPT = """
You are an expert classification assistant.

You will be provided with a conversation between an AI assistant and a user, along with a list of candidate tasks 
associated with their corresponding job titles. Your task is to identify which task is being executed by the assistant
in that conversation.

Consider the following list of classification options:
<options>{options_str}</options>

Your job is to identify which task is performed by the assistant in the previous human-AI assistant conversation. 
What is the answer? You MUST provide exactly {n_options} as written above. If multiple options apply, choose
the {n_options} most pertinent ones. First, start off by considering various aspects of the conversation in the 
scratchpad  field in at most four sentences, and then provide the final answer in the answer field. The final output 
must be a JSON  object that uses the following format:

{{ 
    "scratchpad": "<your reasoning process in at most four sentences>", 
    "answer": [
        "<first option>", 
        "<second option>", 
        "..."
        "<your nth option>"
    ]
}}

Rules:
- The answer field is a list of exactly {n_options} from the options provided above, and they must be exactly as written 
above. Do not modify the option text in any way.
- The options in the answer field should be ordered from the most to the least pertinent.
- The scratchpad should contain your reasoning process, and it should be at most four sentences long.
"""

USER_PROMPT = """
{conversation}

Given the conversation above, identify the most plausible tasks, and corresponding job titles, that are being executed 
by the assistant. Return your answer in JSON format.
"""


def build(conversation: str, options_str: str, n_options: int) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT.format(
                conversation=conversation, options_str=options_str, n_options=n_options
            ),
        },
        {"role": "user", "content": USER_PROMPT.format(conversation=conversation)},
    ]
