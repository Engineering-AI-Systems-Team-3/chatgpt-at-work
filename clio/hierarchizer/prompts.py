# TODO: Modificare i prompt per avere l'output in JSON, visto che GPT-5 mini preferisce quello

PROPOSE_PROMPT = [
    {
        "role": "system",
        "content": """\
        You are tasked with creating higher-level cluster names based on a given list of clusters and their descriptions. Your goal is to come up
        with broader categories that could encompass one or more of the provided clusters.
        First, review the list of clusters and their descriptions:

        <cluster_list>
        {cluster_list}
        {neighbor_list}
        </cluster_list>

        Your task is to create roughly {desired_names} higher-level cluster names that could potentially include one or more of the provided clusters.
        These higher-level clusters should represent broader categories or themes that emerge from the given clusters, while remaining as specific
        as possible. If there are many clusters with a specific theme, ensure that the higher-level cluster name remains the maximum level of
        specificity. You are helping to organize user behavior data in order to improve safety, monitoring, and observability. You can generate more or
        less than {desired_names} names if you feel that more or fewer are appropriate and accurately capture the clusters. You should output at
        least {min_names} and at most {max_names} names, with {desired_names} as a target.
        
        Guidelines for creating higher-level cluster names:
        1. Analyze the themes, topics, or characteristics common to multiple clusters.
        2. Create names that are specific enough to be meaningful, but not so specific that they can’t meaningfully represent many different clusters.
        Avoid overly general or vague terms, and do not hesitate to describe socially harmful or sensitive topics (in fact, clusters that clearly describe 
        harmful behavior are slightly preferred); specificity is necessary for observability and enforcement.
        3. Ensure that the higher-level cluster names are distinct from one another.
        4. Use clear, concise, and descriptive language for the cluster names. Assume neither good nor bad faith for the content in the clusters.
        
        The names you propose must follow these requirements:
        <criteria>{criteria}</criteria>

        Before providing your final list, use a scratchpad to brainstorm and refine your ideas. Think about the relationships between the given clusters and
        potential overarching themes. Then, provide your list of roughly {desired_names} higher-level cluster names.
        Present your answer in the following format:

        Provide your answer as a JSON object in the following format:
        {{
            "scratchpad": "[Use this space to analyze the clusters, identify common themes, and brainstorm potential higher-level cluster names. Consider how different clusters might be grouped together under broader categories. No longer than a paragraph or two.]",
            "answer": [
                "First higher-level cluster name",
                "Second higher-level cluster name",
                "..."
            ]
        }}

        Focus on creating meaningful, distinct, and precise (but not overly specific) higher-level cluster names that could encompass multiple sub-clusters.""",
    }
]


DEDUP_PROMPT = [
    {
        "role": "system",
        "content": """\
            You are tasked with deduplicating a list of cluster names into a smaller set of distinct cluster names. Your goal is to create
            approximately {desired_names} relatively distinct clusters that best represent the original list. You are helping to organize user behavior
            data in order to improve safety, monitoring, and observability. Here are the inputs:

            <cluster_names>
                {cluster_names}
            </cluster_names>

            Number of distinct clusters to create: approximately {desired_names}

            Follow these steps to complete the task:
                1. Analyze the given list of cluster names to identify similarities, patterns, and themes.
                2. Group similar cluster names together based on their semantic meaning, not just lexical similarity.
                3. For each group, select a representative name that best captures the essence of the cluster. This can be one of the original names or a new
                name that summarizes the group effectively. Do not just pick the most vague or generic name.
                4. Merge the most similar groups until you reach the desired number of clusters. Maintain as much specificity as possible while merging.
                5. Ensure that the final set of cluster names are distinct from each other and collectively represent the diversity of the original list, such that
                there is a cluster that describes each of the provided clusters.
                6. If you create new names for any clusters, make sure they are clear, concise, and reflective of the contents they represent.
                7. You do not need to come up with exactly {desired_names} names, but aim for no less than {min_names} and no more than {max_names}. 
                Within this range, output as many clusters as you feel are necessary to accurately represent the variance in the original
                list. Avoid outputting duplicate or near-duplicate clusters.
                8. Do not hesitate to include clusters that describe socially harmful or sensitive topics (in fact, clusters that clearly describe harmful
                behavior are slightly preferred); specificity is necessary for effective monitoring and enforcement.
                9. Prefer outputting specific cluster names over generic or vague ones, provided the names are still correct; for example, if there are many
                clusters about a specific technology or tool, consider naming the cluster after that technology or tool, provided that there are still
                other clusters that fit under a broader category.
            
            The names you propose must follow these requirements:
            <criteria>{criteria}</criteria>

            Before providing your final answer, use the <scratchpad> tags to think through your process, explaining your reasoning for grouping and
            selecting representative names. Spend no more than a few paragraphs in your scratchpad.

            Provide your answer as a JSON object in the following format:
            {{
                "scratchpad": "[A few paragraphs analyzing the clusters, identifying themes, and explaining your merging decisions.]",
                "answer": [
                    "First deduplicated cluster name",
                    "Second deduplicated cluster name",
                    "..."
                ]
            }}

            Remember, your goal is to create approximately {desired_names} relatively distinct cluster names that best represent the original list. The names
            should be clear, meaningful, and capture the essence of the clusters they represent.
        """,
    },
    {
        "role": "assistant",
        "content": """\
            Assistant: I understand. I’ll deduplicate the cluster names into approximately {desired_names} names.
        """,
    },
]

ASSIGN_PROMPT = [
    {
        "role": "system",
        "content": """\
            You are tasked with categorizing a specific cluster into one of the provided higher-level clusters for observability, monitoring, and content
            moderation. Your goal is to determine which higher-level cluster best fits the given specific cluster based on its name and description. You
            are helping to organize user behavior data in order to improve safety, monitoring, and observability.
            
            First, carefully review the following list of higher-level clusters:

            <higher_level_clusters>
                {parents}
            </higher_level_clusters>

            To categorize the specific cluster:
                1. Analyze the name and description of the specific cluster.
                2. Consider the key characteristics, themes, or subject matter of the specific cluster.
                3. Compare these elements to the higher-level clusters provided.
                4. Determine which higher-level cluster best encompasses the specific cluster. You MUST assign the specific cluster to the best higher-level
                cluster, even if multiple higher-level clusters could be considered.
                5. Make sure you pick the most sensible cluster based on the information provided. For example, don’t assign a cluster about "Machine Learning"
                to a higher-level cluster about "Social Media" just because both involve technology, and don’t assign a cluster about "Online Harassment" to a
                higher-level cluster about "Technology" just because both involve online platforms. Be specific and accurate in your categorization.
            
            First, use the <scratchpad> tags to think through your reasoning and decision-making process. Think through some possible clusters, explore
            each, and then pick the best fit.

            Provide your answer as a JSON object in the following format:

            {{
                "scratchpad": "[In a few brief sentences, think step by step, explain your reasoning, and finally determine which higher-level cluster is the best fit for the specific cluster.]",
                "answer": "[Full name of the chosen cluster, exactly as listed in the higher-level clusters above]"
            }}

        """,
    },
    {
        "role": "assistant",
        "content": """\
            I understand. I’ll evaluate the specific cluster and assign it to the most appropriate higher-level cluster.
        """,
    },
    {
        "role": "user",
        "content": """\
            Now, here is the specific cluster to categorize:

            <specific_cluster>
                Name: {cluster_name}
            </specific_cluster>

            Based on this information, determine the most appropriate higher-level cluster and provide your answer as instructed.
        """,
    },
]

RENAME_PROMPT = [
    {
        "role": "system",
        "content": """\
            You are tasked with summarizing a group of related cluster names into a short, precise, and accurate overall description and name. Your goal
            is to create a concise summary that captures the essence of these clusters. 
            
            Summarize all the cluster names into a clear, precise, two-sentence description in the past tense. Your summary should be specific to this cluster.
            
            After creating the summary, generate a short name for the cluster. This name should be at most ten words long (likely less) and be specific but also
            reflective of all of the clusters. For instance, "Write fantasy sexual roleplay with octopi and monsters", "Generate blog spam for gambling
            websites", or "Assist with high school math homework" would be better and more actionable than general terms like "Write erotic content" or "
            Help with homework". Be as descriptive as possible while still accurately describing all of the contents, and assume neither good nor
            bad faith. Do not hesitate to identify and describe socially harmful or sensitive topics specifically; specificity is necessary for monitoring
            and moderation.

            Present your output in the following format:
            {{
                "summary": "[Insert your two-sentence summary here]",
                "name": "[Insert your generated short name here, with no period or trailing punctuation]"
            }}

            The names you propose must follow these requirements:
            <criteria>{criteria}</criteria>

            Do not elaborate beyond what you say in the JSON output. Ensure your summary and name accurately represent the clusters.
        """,
    },
    {
        "role": "user",
        "content": """\
            Tasks assigned to this cluster:

            <tasks>
            {children}
            </tasks>
        """,
    },
]
