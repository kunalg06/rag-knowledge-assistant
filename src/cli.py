import sys
from src.rag_chain import answer_question


def main():
    if len(sys.argv) < 2:
        print("Usage: python src/cli.py \"your question here\"")
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    print(f"Question: {query}\n")

    result = answer_question(query)
    print("Answer:\n")
    print(result["answer"])
    print("\nSources used:")
    for src in result["sources"]:
        print(f"- {src}")


if __name__ == "__main__":
    main()
