# chatgpt-at-work

To study how artificial intelligence may reshape work, Anthropic mapped Claude queries to occupational tasks to estimate which tasks are most exposed to automation by LLMs. It is not clear whether the same patterns hold for ChatGPT.

We test this by mapping millions of publicly available ChatGPT interaction logs [1] to tasks and skills in the Occupational Information Network (O\*NET), including detailed work activities (DWAs). We:

- map ChatGPT interactions to O\*NET tasks and DWAs;
- map the same interactions to the roles a chatbot can take (for example, friend, colleague, or trainer), using the role set in [2];
- evaluate both mappings with a human-in-the-loop protocol, with a focus on separating plausible automation targets from spurious matches; and
- release a dataset and analysis code so others can reproduce our results.
  References

[1] WildChat Datasets: 4.8M and 50M

[2] Frictionless Love: Associations Between AI Companion Roles and Behavioral Addiction
