#!/usr/bin/env python3
"""
generate_qa.py — Stage 3 QA generation, Steps 2-4: LLM caller, validator,
question-type classifier. Dataset-agnostic (takes plain text in, no
dataset-specific logic here).

Uses the OpenAI SDK (key from .env / OPENAI_API_KEY) — this is what was
already installed and working in this environment, per the pass-1 spec.
"""
import json
import os
import re
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL = 'gpt-4o-mini'
SLEEP_SECONDS = 0.5

SYSTEM_PROMPT = (
    "You generate factual QA pairs for a retrieval benchmark. Given a "
    "text passage, produce one specific factual question and its answer. "
    "Requirements: the question must be answerable ONLY from the provided "
    "text, not from general world knowledge. The answer must be 1-2 "
    "sentences maximum. "
    "The question must NOT be answerable from the title or first sentence "
    "alone. It should require understanding a specific detail, nuance, "
    "method, result, or limitation discussed in the body of the text. "
    "Avoid questions that begin with 'What does this paper propose' or "
    "'What is the main contribution of'. Prefer questions about specific "
    "findings, constraints, experimental conditions, comparisons, or "
    "failure modes. "
    "Your response must be valid JSON. If the text contains LaTeX notation, "
    "backslashes, or mathematical symbols, do not include them in your "
    "question or answer — paraphrase using plain English instead. Never "
    "use literal backslashes in your JSON output. "
    "Return valid JSON only, no other text: "
    '{"question": "...", "answer": "..."}'
)

AGGREGATE_SYSTEM_PROMPT = (
    "You are generating evaluation questions for a graph retrieval "
    "benchmark. Given multiple research paper abstracts from the same "
    "field, generate one question whose correct answer requires "
    "synthesizing information across at least three of the provided "
    "papers. The question should NOT be answerable from any single paper "
    "alone. Good question types: dominant methods across the field, "
    "common limitations mentioned by multiple authors, recurring "
    "experimental setups, contrasting approaches to the same problem. "
    "If the text contains LaTeX notation, backslashes, or mathematical "
    "symbols, do not include them in your question or answer — paraphrase "
    "using plain English instead. Never use literal backslashes in your "
    "JSON output. "
    'Return JSON only: {"question": "...", "answer": "..."}'
)

# Control characters that indicate JSON-escape corruption (a literal
# backslash from source LaTeX consumed as \n \f \t \r during json.loads,
# rather than being escaped by the model as \\n \\f \\t \\r).
_CONTROL_CHAR_RE = re.compile(r'[\x0c\x08\t\r]')

REFUSAL_PHRASES = ["I cannot", "As an AI", "I don't have", "I'm sorry"]


def _client() -> OpenAI:
    key = os.environ.get('OPENAI_API_KEY')
    if not key:
        raise RuntimeError('OPENAI_API_KEY not set (checked .env / environment)')
    return OpenAI(api_key=key)


def _call_llm_raw(system_prompt: str, user_content: str, client: OpenAI = None) -> str:
    """Low-level call, returns the raw (unparsed) response string. Exposed for
    debugging JSON-corruption cases where we need to see what the model
    actually returned before parsing."""
    client = client or _client()
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_content},
            ],
            temperature=0.3,
            response_format={'type': 'json_object'},
        )
        return resp.choices[0].message.content
    except Exception as e:
        raise ValueError(f'API call failed: {type(e).__name__}: {e}')


def generate_qa(text: str, client: OpenAI = None) -> dict:
    """
    Calls the LLM on a single text passage. Returns {"question": str, "answer": str}.
    Raises ValueError if the API call fails or the response cannot be parsed.
    """
    raw = _call_llm_raw(SYSTEM_PROMPT, f'Text: {text}', client=client)

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f'Could not parse response as JSON: {e}\nraw={raw!r}')

    if 'question' not in parsed or 'answer' not in parsed:
        raise ValueError(f'Response JSON missing question/answer keys: {parsed!r}')

    time.sleep(SLEEP_SECONDS)
    return parsed


def generate_aggregate_qa(rows: list[dict], client: OpenAI = None) -> dict:
    """
    Takes a list of 5-10 rows (dicts with at least 'text_fidelity_b' and
    'text_fidelity_a') from the SAME categorical_label/aggregate_id and
    generates one question answerable only by synthesizing across >=3 of them.
    Returns {"question": str, "answer": str}. Raises ValueError on failure.
    """
    listed = '\n'.join(
        f'Paper {i+1}: {r.get("text_fidelity_b", "")} — {r.get("text_fidelity_a", "")}'
        for i, r in enumerate(rows)
    )
    raw = _call_llm_raw(AGGREGATE_SYSTEM_PROMPT, listed, client=client)

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f'Could not parse response as JSON: {e}\nraw={raw!r}')

    if 'question' not in parsed or 'answer' not in parsed:
        raise ValueError(f'Response JSON missing question/answer keys: {parsed!r}')

    time.sleep(SLEEP_SECONDS)
    return parsed


def validate_qa(qa_dict: dict) -> tuple[bool, str]:
    """
    Checks, in order, first failure wins:
      - question/answer keys exist
      - question is a string with > 15 words
      - answer is a string with > 5 words
      - question ends with '?'
      - neither field contains a refusal phrase
      - neither field contains a stray control character (form feed,
        backspace, tab, carriage return) — a tell for JSON-escape corruption
        from unescaped LaTeX backslashes (\\n \\f \\t \\r consumed as JSON
        escapes during parsing rather than literal backslash+letter)
    """
    if 'question' not in qa_dict or 'answer' not in qa_dict:
        return False, 'missing question/answer key'

    q, a = qa_dict['question'], qa_dict['answer']

    if not isinstance(q, str) or len(q.split()) <= 10:
        return False, f'question too short ({len(q.split()) if isinstance(q, str) else "n/a"} words, need >10)'

    if not isinstance(a, str) or len(a.split()) <= 5:
        return False, f'answer too short ({len(a.split()) if isinstance(a, str) else "n/a"} words, need >5)'

    if not q.strip().endswith('?'):
        return False, 'question does not end with ?'

    for phrase in REFUSAL_PHRASES:
        if phrase.lower() in q.lower() or phrase.lower() in a.lower():
            return False, f'refusal phrase detected: "{phrase}"'

    if _CONTROL_CHAR_RE.search(q) or _CONTROL_CHAR_RE.search(a):
        return False, 'control_chars'

    return True, 'valid'


def sample_category_groups(df, n_groups: int, group_size_range=(5, 10), seed=42) -> list:
    """
    Groups rows by aggregate_id (== categorical_label for arxiv), keeps
    groups with >= group_size_range[0] members, and samples n_groups of them
    (group_size_range[1] rows each) for aggregate question generation.
    Returns a list of lists-of-row-dicts.
    """
    import random
    lo, hi = group_size_range
    groups = df.groupby('aggregate_id')
    candidates = [g for _, g in groups if len(g) >= lo]

    rng = random.Random(seed)
    rng.shuffle(candidates)

    out = []
    for g in candidates[:n_groups]:
        sample = g.sample(n=min(hi, len(g)), random_state=seed)
        out.append(sample.to_dict('records'))
    return out


def sample_cross_category_groups(df, n_groups: int, group_size_range=(5, 10),
                                   min_categories=3, max_resamples=20, seed=42) -> list:
    """
    Samples n_groups groups of 5-10 rows EACH SPANNING >= min_categories
    distinct categorical_label values (the inverse of sample_category_groups,
    which deliberately groups by a single shared category). Same-category
    groups let retrieval get lucky (2/5 in the gate scored context_relevance
    =1.0); cross-category groups are harder to cover with top-3 retrieval.

    Returns a list of lists-of-row-dicts. Each attempt resamples up to
    max_resamples times if the 3-category minimum isn't met; groups that
    never meet it within the budget are skipped (not padded/faked).
    """
    lo, hi = group_size_range
    rng_seed = seed
    out = []
    for i in range(n_groups):
        found = None
        for attempt in range(max_resamples):
            size = lo + (attempt % (hi - lo + 1))
            sample = df.sample(n=min(size, len(df)), random_state=rng_seed)
            rng_seed += 1
            if sample['aggregate_id'].nunique() >= min_categories:
                found = sample
                break
        if found is not None:
            out.append(found.to_dict('records'))
    return out


def classify_question_type(question: str) -> str:
    """Pure heuristic, no API call. relational / categorical / descriptive."""
    q = question.lower()

    relational_words = ['who', 'whose', 'between', 'related', 'connected',
                         'collaboration', 'co-author', 'coauthor']
    categorical_words = ['what type', 'which category', 'what kind',
                          'what field', 'what domain']

    if any(w in q for w in relational_words):
        return 'relational'
    if any(w in q for w in categorical_words):
        return 'categorical'
    return 'descriptive'


if __name__ == '__main__':
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from format_rows import format_arxiv_row

    data_path = Path(__file__).resolve().parent.parent.parent / 'data/arxiv/train/samples/sample_00.jsonl'
    rows = []
    with open(data_path, encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            text = format_arxiv_row(row)
            if text:
                rows.append((row, text))
            if len(rows) >= 5:
                break

    client = _client()
    step2_outputs = []
    print('=' * 20, 'STEP 2 — LLM CALLER', '=' * 20)
    for i, (row, text) in enumerate(rows):
        print(f'\n--- row {i} (primary_id={row["primary_id"]}) ---')
        print(f'text preview: {text[:150]}...')
        try:
            qa = generate_qa(text, client=client)
            print(f'parsed output: {json.dumps(qa, indent=2)}')
            step2_outputs.append(qa)
        except ValueError as e:
            print(f'FAILED: {e}')
            step2_outputs.append(None)

    n_ok = sum(1 for x in step2_outputs if x is not None)
    print(f'\nStep 2: {n_ok}/5 returned valid JSON')

    print('\n' + '=' * 20, 'STEP 3 — VALIDATOR', '=' * 20)
    step3_results = []
    for i, qa in enumerate(step2_outputs):
        if qa is None:
            print(f'row {i}: SKIPPED (step 2 failed)')
            step3_results.append(None)
            continue
        valid, reason = validate_qa(qa)
        print(f'row {i}: ({valid}, {reason!r})  Q: {qa["question"]!r}')
        step3_results.append((valid, reason))

    print('\n' + '=' * 20, 'STEP 4 — QUESTION TYPE CLASSIFIER', '=' * 20)
    for i, qa in enumerate(step2_outputs):
        if qa is None or not step3_results[i] or not step3_results[i][0]:
            continue
        qtype = classify_question_type(qa['question'])
        print(f'[{qtype:<12}] {qa["question"]}')
