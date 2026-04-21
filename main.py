import json
import os
import re
import string
import time

import pandas as pd
import torch

torch.set_default_device('cpu')

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils.evaluate_results import NO_ANSWER_MARKER, evaluate_results

transformers.utils.logging.set_verbosity_error()

SYSTEM_PROMPT = """Extract the exact verbatim answer to the Question from the Context.
Output NO ANSWER if the answer is missing, impossible, or requires assumptions.

FORMAT:
Reasoning: [1 sentence logical check]
Confidence: [1-10]
Final Answer: [exact context span] OR NO ANSWER

EXAMPLE:
Context: The sun rises in the east.
Question: Where does the sun set?
Reasoning: The context only mentions where it rises, not where it sets.
Confidence: 10
Final Answer: NO ANSWER"""



# 1. CPU parallelization: use all available cores
num_cpus = os.cpu_count() or 1
torch.set_num_threads(num_cpus)
torch.set_num_interop_threads(num_cpus)
print(f"[HW] Using {num_cpus} CPU threads")

# 2. dtype: prefer bfloat16 if hardware supports it, fall back to float32
#    bfloat16 on CPU requires AVX512-BF16 (Intel Cooperlake+) or ARM with BF16
try:
    _ = torch.zeros(1, dtype=torch.bfloat16) + torch.zeros(1, dtype=torch.bfloat16)
    dtype = torch.bfloat16
    print("[HW] bfloat16 supported — using torch.bfloat16")
except Exception:
    dtype = torch.float32
    print("[HW] bfloat16 not supported — falling back to torch.float32")

model_name = 'meta-llama/Llama-3.2-3B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(model_name, token=True)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, token=True)
model.config.pad_token_id = tokenizer.eos_token_id
tokenizer.pad_token_id = tokenizer.eos_token_id


def query(messages: list[dict], max_new_tokens: int = 80) -> str:
    """Run a chat-formatted message list through the model and return the reply text.

    Args:
        messages (list[dict]): A list of message dictionaries for the chat template.
            Each dictionary typically requires 'role' and 'content' keys.
        max_new_tokens (int, optional): The maximum number of tokens to generate.
            Defaults to 80.

    Returns:
        str: The generated reply text from the model, stripped of special tokens
            and leading/trailing whitespace.
    """
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_dict=True,
        return_tensors='pt', padding=True, truncation=True
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    outputs = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=False,
    )

    new_tokens = outputs[0][inputs['input_ids'].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def normalize_text(text: str) -> str:
    """Normalize text by lowercasing, removing punctuation, and stripping extra whitespace.

    Args:
        text (str): The input text to normalize.

    Returns:
        str: The normalized text.
    """
    text = text.lower()
    text = ''.join(ch for ch in text if ch not in string.punctuation)
    return ' '.join(text.split())


def retrieve_top_sentences(context: str, question: str, max_n: int = 3, threshold_ratio: float = 0.5) -> str:
    """Return the most relevant context sentences for the question via TF-IDF.

    Instead of always taking a fixed top-N, we take up to max_n sentences but
    drop any whose score is below (threshold_ratio * best_score). This means:
      - If 2 sentences are strong and the 3rd is weak, only 2 are returned.
      - If only 1 sentence is clearly relevant, only 1 is returned.
      - Always returns at least 1 sentence (the best one).

    Args:
        context (str): The full context paragraph to extract sentences from.
        question (str): The specific question being asked.
        max_n (int, optional): The maximum number of sentences to return. Defaults to 3.
        threshold_ratio (float, optional): A sentence must score at least this ratio
            of the best sentence's score to be included. Defaults to 0.5.

    Returns:
        str: A single string containing the most relevant sentences joined by spaces,
            preserving their original order from the context.
    """
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', context) if s.strip()]
    if len(sentences) <= max_n:
        return context  # short context: use it all

    try:
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform([question] + sentences)
        scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]

        best_score = scores.max()

        if best_score < 0.05:
            return ""

        # Always include at least the best sentence; drop anything below the ratio
        cutoff = best_score * threshold_ratio
        top_indices = [
            i for i in scores.argsort()[-max_n:][::-1]
            if scores[i] >= cutoff
        ]

        # Preserve original reading order
        top_indices_sorted = sorted(top_indices)
        return ' '.join(sentences[i] for i in top_indices_sorted)
    except Exception:
        return context


def judge(citation: str, question: str, confidence_threshold: int = 7) -> str:
    """Prompt the LLM to extract an exact answer or return NO ANSWER.

    Uses CoT reasoning to catch the trap types documented in the SQuAD 2.0 paper.

    Args:
        citation (str): The retrieved context sentences to search for the answer.
        question (str): The question to be answered.
        confidence_threshold (int, optional): The minimum confidence score (1-10)
            required to return a valid answer instead of NO ANSWER. Defaults to 7.

    Returns:
        str: The extracted exact answer span from the context, or 'NO ANSWER'.
    """
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': f"Context: {citation}\n\nQuestion: {question}"},
    ]

    raw_response = query(messages, max_new_tokens=60)

    # Parse confidence and final answer
    match = re.search(
        r'Confidence:\s*(\d+).*?Final Answer:\s*(.*)',
        raw_response, flags=re.IGNORECASE | re.DOTALL
    )

    if match:
        confidence_score = int(match.group(1))
        final_answer = match.group(2).strip()
        final_answer = final_answer.replace('**', '').replace('"', '').replace("'", '').strip()
        final_answer = final_answer.split('\n')[0].strip()

        if not final_answer: # LLM failed to return response in the expected format
            final_answer = 'NO ANSWER'
        elif confidence_score < confidence_threshold and final_answer.lower() != 'no answer':
            final_answer = 'NO ANSWER'
    else:
        # Fallback: scan raw output for any NO ANSWER
        if re.search(r'no\s+answer', raw_response, re.IGNORECASE):
            final_answer = 'NO ANSWER'
        else:
            final_answer = 'NO ANSWER'  # safe default — never hallucinate

    # Substring enforcement: extracted answer MUST appear verbatim in context
    if final_answer.upper() != 'NO ANSWER':
        norm_ans = normalize_text(final_answer)
        norm_ctx = normalize_text(citation)
        if norm_ans not in norm_ctx:
            final_answer = 'NO ANSWER'

    return final_answer


def squad_qa(data_filename: str) -> str:
    """Run a two-stage QA pipeline on a SQuAD 2.0 dataset.

    Stage 1: TF-IDF retrieves the top-3 most relevant sentences (no LLM).
    Stage 2: LLM judge evaluates those 3 sentences to extract an answer or output NO ANSWER.

    Args:
        data_filename (str): The file path to the input CSV containing SQuAD 2.0 data.
            The CSV must contain 'context', 'question', and 'is_impossible' columns.

    Returns:
        str: The file path to the output CSV containing the predictions in a
            new 'final answer' column.
    """
    df = pd.read_csv(data_filename)
    final_answers = []

    CONFIDENCE_THRESHOLD = 7

    for idx, row in df.iterrows():
        context       = str(row['context'])
        question      = str(row['question'])

        # Stage 1: retrieve only meaningfully relevant sentences
        citation = retrieve_top_sentences(context, question, max_n=3, threshold_ratio=0.3)

        # Stage 2: LLM judge (skip if context is completely irrelevant)
        if not citation:
            raw_answer = 'NO ANSWER'
        else:
            raw_answer = judge(citation, question, confidence_threshold=CONFIDENCE_THRESHOLD)

        answer = NO_ANSWER_MARKER if 'NO ANSWER' in raw_answer.upper() else raw_answer
        final_answers.append(answer)


    df['final answer'] = final_answers
    out_filename = data_filename.replace('.csv', '-results.csv')
    df.to_csv(out_filename, index=False)

    print(f'Final answers recorded into {out_filename}')
    return out_filename


if __name__ == '__main__':
    start_time = time.time()

    with open('config.json', 'r') as json_file:
        config = json.load(json_file)

    data = pd.read_csv(config['data'])
    sample = data.sample(n=config['sample_for_solution'])  # for grading will be replaced with 'sample_for_grading'
    sample_filename = config['data'].replace('.csv', '-sample.csv')
    sample.to_csv(sample_filename, index=False)

    out_filename = squad_qa(sample_filename)  # todo: the function you implement

    eval_out = evaluate_results(out_filename, final_answer_column='final answer')
    eval_out_list = [str((k, round(v, 3))) for (k, v) in eval_out.items()]
    print('\n'.join(eval_out_list))

    elapsed_time = time.time() - start_time
    print(f"time: {elapsed_time: .2f} sec")
