import sys
from cache import init_db
from harness import run_harness


def main():
    init_db()
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
    else:
        prompt = "Return ONLY valid JSON with keys 'language' and 'year' for Python."

    print("=" * 60)
    print("Agentic Harness: Reliability & Semantic Caching Framework")
    print("=" * 60)
    print(f"Prompt: {prompt}\n")

    try:
        result = run_harness(prompt, use_cache=True)
        print("Result:")
        print(f"- Answer: {result['answer']}")
        print(f"- Initial Score: {result['initial_score']:.2f}")
        print(f"- Final Reliability: {result['final_score']:.2f}")
        print(f"- Retries: {result['retries']}")
        print(f"- Cached: {result['cached']}")
        print(f"- Latency: {result['latency_ms']} ms")
        print("\nExecution Trace:")
        for step in result["trace"]:
            print(f"  {step}")
    except Exception as e:
        print(f"Error running harness: {e}")
        print("\nNote: Ensure a valid GROQ_API_KEY (starts with 'gsk_') is set in your .env file.")


if __name__ == "__main__":
    main()
