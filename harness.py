import os
import time
import json
import requests
from dotenv import load_dotenv
from cache import check_cache, save_to_cache, log_execution

load_dotenv()
API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = MODEL = "openai/gpt-oss-20b"

def call_llm(messages: list) -> str:
    """Sends a conversation history to Groq's Llama 3.1 8B model and returns text."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is not set. Please add a valid GROQ_API_KEY to your .env file."
        )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.3  # Low temperature = more focused, less random
    }
    response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
    data = response.json()
    if "choices" not in data:
        error_msg = data.get("error", {}).get("message", str(data))
        raise RuntimeError(
            f"Groq API Error: {error_msg}. Please check that your GROQ_API_KEY in .env is a valid Groq key (usually starts with 'gsk_')."
        )
    return data["choices"][0]["message"]["content"]

def deterministic_validate(prompt: str, answer: str) -> tuple[bool, str]:
    """Rule-based checks that require zero AI."""
    # Rule 1: Must not be empty or tiny
    if len(answer.strip()) < 15:
        return False, "Answer is too short (under 15 characters)."
    
    # Rule 2: If the prompt explicitly asks for JSON, verify it parses cleanly
    if "json" in prompt.lower():
        try:
            start, end = answer.find("{"), answer.rfind("}") + 1
            if start == -1 or end == 0:
                return False, "Prompt requested JSON, but no JSON object {} was found."
            json.loads(answer[start:end])
        except json.JSONDecodeError:
            return False, "Output contains invalid JSON syntax."
            
    return True, "Passed deterministic rules."

def critic_evaluate(prompt: str, answer: str) -> tuple[float, str]:
    """LLM-as-a-Judge: Grades the answer from 0.0 to 1.0 and gives actionable feedback."""
    critic_instruction = f"""
    You are a strict quality-assurance critic. Evaluate the Answer against the Prompt.
    Check for accuracy, completeness, and adherence to all constraints in the prompt.
    Respond ONLY with a valid JSON object in this exact format:
    {{"score": 0.85, "feedback": "Brief explanation of what is good or missing"}}

    Prompt: {prompt}
    Answer: {answer}
    """
    raw_output = call_llm([{"role": "user", "content": critic_instruction}])
    try:
        start = raw_output.find("{")
        end = raw_output.rfind("}") + 1
        parsed = json.loads(raw_output[start:end])
        score = float(parsed.get("score", 0.5))
        feedback = str(parsed.get("feedback", "No feedback provided."))
        return max(0.0, min(1.0, score)), feedback
    except Exception:
        return 0.5, "Critic failed to output valid JSON evaluation."

def run_harness(prompt: str, max_retries: int = 2, use_cache: bool = True) -> dict:
    """Main orchestration pipeline."""
    start_time = time.time()

    # STEP 1: Check Semantic Cache
    if use_cache:
        cached = check_cache(prompt)
        if cached:
            latency_ms = round((time.time() - start_time) * 1000, 2)
            log_execution(prompt, cached["reliability"], cached["reliability"], 0, True, latency_ms)
            return {
                "answer": cached["response"],
                "initial_score": cached["reliability"],
                "final_score": cached["reliability"],
                "retries": 0,
                "cached": True,
                "similarity": cached["similarity"],
                "latency_ms": latency_ms,
                "trace": [f"Cache HIT (Similarity: {cached['similarity']}) -> Returned in {latency_ms} ms"]
            }

    # STEP 2: Iterative Generation + Validation + Retry Loop
    messages = [{"role": "user", "content": prompt}]
    retries = 0
    initial_score = 0.0
    final_score = 0.0
    answer = ""
    trace = []

    while retries <= max_retries:
        answer = call_llm(messages)
        
        # Run Deterministic Validation first
        passed_rules, rule_feedback = deterministic_validate(prompt, answer)
        if not passed_rules:
            score = 0.30
            feedback = rule_feedback
        else:
            # Run Critic Evaluation
            score, feedback = critic_evaluate(prompt, answer)

        if retries == 0:
            initial_score = score
        final_score = score
        trace.append(f"Attempt {retries + 1}: Score = {score:.2f} | Feedback: {feedback}")

        # Stop early if reliability threshold (>= 0.80) is met!
        if score >= 0.80:
            break

        # Otherwise, append feedback and retry
        if retries < max_retries:
            messages.append({"role": "assistant", "content": answer})
            messages.append({
                "role": "user",
                "content": f"Your previous response failed quality checks (Score: {score:.2f}). Fix it based on this feedback: {feedback}"
            })
        retries += 1

    latency_ms = round((time.time() - start_time) * 1000, 2)

    # STEP 3: Save to cache if reliable
    if use_cache:
        save_to_cache(prompt, answer, final_score)
        log_execution(prompt, initial_score, final_score, retries, False, latency_ms)

    return {
        "answer": answer,
        "initial_score": initial_score,
        "final_score": final_score,
        "retries": retries,
        "cached": False,
        "similarity": 0.0,
        "latency_ms": latency_ms,
        "trace": trace
    }


if __name__ == "__main__":
    import sys
    test_prompt = sys.argv[1] if len(sys.argv) > 1 else "Return valid JSON with keys 'language' and 'year' for Python."
    print(f"Running Agentic Harness with prompt:\n\"{test_prompt}\"\n")
    try:
        result = run_harness(test_prompt, use_cache=True)
        print("Harness Result:")
        print(f"- Answer: {result['answer']}")
        print(f"- Initial Score: {result['initial_score']}")
        print(f"- Final Score: {result['final_score']}")
        print(f"- Retries: {result['retries']}")
        print(f"- Cached: {result['cached']}")
        print(f"- Latency: {result['latency_ms']} ms")
        print("\nExecution Trace:")
        for step in result["trace"]:
            print(f"  {step}")
    except Exception as e:
        print(f"Execution Error: {e}")
