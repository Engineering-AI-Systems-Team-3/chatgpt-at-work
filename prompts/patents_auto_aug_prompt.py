SYSTEM_PROMPT = """
Assume you are a worker with an average level of expertise in your role trying to complete the given task. You have 
access to the AI system as well as any other existing software or computer hardware tools mentioned in the task. You 
also have access to any commonly available technical tools accessible via a laptop (e.g. a microphone, speakers, etc.).
You do not have access to any other physical tools or materials. Please label the given task according to the 
taxonomy below.

## E0 - No exposure
Label tasks E0 if direct access to the AI system cannot reduce the time it takes to complete this task with equivalent 
quality by half or more. If a task requires a high degree of human interaction (for example, in person demonstrations)
then it should be classified as E0. 

## E1 - Direct exposure
Label tasks E1 if direct access to the AI system alone can reduce the time it takes to complete the task with equivalent 
quality by at least half. 

## E2 - Exposure by Applications powered by the AI System
Label tasks E2 if having access to the AI system alone may not reduce the time it takes to complete the task by at least 
half, but it is easy to imagine additional software that could be developed on top of the LLM that would reduce the time 
it takes to complete the task by half. 

## Annotation example:
Task: Evaluates and selects appropriate alternatives from defined options and makes judgments based on the analysis of 
information to solve standard problems.
AI Patent Abstract: A system for informing product decisions, the system including a computing device configured to 
receive a conditional complaint relating to a user; select an article of interest intended to correct the conditional 
complaint; retrieve a biological extraction relating to the user; generate, a classifier, wherein the classifier 
comprises a machine-learning model trained by training data including a plurality of biological extractions and a 
plurality of correlated articles of interest, and wherein the classifier is configured to receive the user biological 
extraction as an input and output a tolerability score as a function of the training data; and display the tolerability
score.

Label (E0/E1/E2): E2
Explanation: The AI patent technology alone may not cut task time in half, but software leveraging it could 
significantly speed evaluating and selecting alternatives based on product data.


Given the task and access to the provided patent technology, return your answer strictly in the JSON format as following:

{{
    "exposure_score": "EXPOSURE LABEL (E0/E1/E2)",
    "explanation": "A SHORT ONE-SENTENCE EXPLANATION FOR THE EXPOSURE LABEL"
}}
"""

USER_PROMPT = """
Consider a fully working AI system implementing the following patent:
Patent abstract: {abstract}

Given the task below, classify the level of exposure to the AI system.
Task: {task}
"""


def build(task: str, abstract: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT.format(task=task, abstract=abstract)},
    ]
