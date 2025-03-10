"""
Usage:
python3 local_example_complete.py
"""

import sglang as sgl


@sgl.function
def few_shot_qa(s, question):
    s += """The following are questions with answers.
Q: What is the capital of France?
A: Paris
Q: What is the capital of Germany?
A: Berlin
Q: What is the capital of Italy?
A: Rome
"""
    s += "Q: " + question + "\n"
    s += "A:" + sgl.gen("answer", stop="\n", temperature=0)


def single():
    state = few_shot_qa.run(question="What is the capital of the United States?")
    answer = state["answer"].strip().lower()

    assert "washington" in answer, f"answer: {state['answer']}"

    print(state.text())


def stream():
    state = few_shot_qa.run(
        question="What is the capital of the United States?", stream=True
    )

    for out in state.text_iter("answer"):
        print(out, end="", flush=True)
    print()


def batch():
    states = few_shot_qa.run_batch(
        [
            {"question": "What is the capital of the United States?"},
            {"question": "What is the capital of China?"},
        ]
    )

    for s in states:
        print(s["answer"])


if __name__ == "__main__":
    runtime = sgl.Runtime(model_path="/home/cjmcv/project/llm_models/Qwen/Qwen2___5-1___5B-Instruct")#, disable_cuda_graph=True
    sgl.set_default_backend(runtime)

    # Run a single request
    print("\n========== single ==========\n")
    single()

    # Stream output
    print("\n========== stream ==========\n")
    stream()

    # Run a batch of requests
    print("\n========== batch ==========\n")
    batch()

    runtime.shutdown()
