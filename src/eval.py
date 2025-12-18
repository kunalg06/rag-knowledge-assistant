import os
import json

from rag_chain import answer_question

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
EVAL_FILE = os.path.join(BASE_DIR, "data", "eval.jsonl")


def load_eval_data():
    if not os.path.exists(EVAL_FILE):
        raise FileNotFoundError(f"Eval file not found at {EVAL_FILE}")

    examples = []
    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            examples.append(json.loads(line))
    return examples


def keyword_metrics(answer: str, keywords):
    answer_lower = answer.lower()
    total = len(keywords)
    if total == 0:
        return 0.0, 0.0

    hit = 0
    for kw in keywords:
        if kw.lower() in answer_lower:
            hit += 1

    precision = hit / len(answer.split()) if answer.split() else 0.0
    recall = hit / total
    return precision, recall


def main():
    examples = load_eval_data()
    all_precisions = []
    all_recalls = []

    for ex in examples:
        question = ex["question"]
        keywords = ex.get("expected_keywords", [])

        print(f"Question: {question}")
        result = answer_question(question)
        answer = result["answer"]
        print(f"Answer: {answer}\n")

        precision, recall = keyword_metrics(answer, keywords)
        all_precisions.append(precision)
        all_recalls.append(recall)

        print(f"Expected keywords: {keywords}")
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}")
        print("-" * 40)

    if all_precisions:
        avg_p = sum(all_precisions) / len(all_precisions)
        avg_r = sum(all_recalls) / len(all_recalls)
        print(f"\nAverage Precision: {avg_p:.4f}")
        print(f"Average Recall: {avg_r:.4f}")


if __name__ == "__main__":
    main()
