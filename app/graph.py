from langgraph.graph import END, StateGraph

from state import GraphState
from nodes import retrieve, grade_documents, web_search, generate, grade_generation

MAX_GENERATIONS = 2  # critic may trigger at most one regeneration


# --- ROUTING LOGIC ---
def decide_to_generate(state: GraphState):
    """Reads the 'web_search' flag set by the Grader node."""
    print("[Router] Deciding next step...")
    if state["web_search"] == "Yes":
        print("[Router] Decision: Chunks are irrelevant. Routing to Web Search.")
        return "web_search"
    print("[Router] Decision: Chunks are good. Skipping web search -> Routing to Generate.")
    return "generate"


def decide_after_critic(state: GraphState):
    """End if the answer is grounded or the retry cap is hit; otherwise regenerate."""
    if state["grounded"] == "yes":
        print("[Router] Answer is grounded. Finishing.")
        return "end"
    if state["retries"] >= MAX_GENERATIONS:
        print("[Router] Retry cap reached. Returning best available answer.")
        return "end"
    print("[Router] Answer not grounded. Regenerating.")
    return "regenerate"


# --- BUILD THE GRAPH ---
workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("web_search", web_search)
workflow.add_node("generate", generate)
workflow.add_node("grade_generation", grade_generation)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_documents")

workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {"web_search": "web_search", "generate": "generate"},
)

workflow.add_edge("web_search", "generate")
workflow.add_edge("generate", "grade_generation")

workflow.add_conditional_edges(
    "grade_generation",
    decide_after_critic,
    {"end": END, "regenerate": "generate"},
)

crag_app = workflow.compile()


def initial_state(question: str) -> GraphState:
    return {
        "question": question,
        "documents": [],
        "web_search": "",
        "generation": "",
        "grounded": "",
        "retries": 0,
    }


# --- TERMINAL TEST EXECUTION ---
if __name__ == "__main__":
    print("\n--- STARTING CRAG EXECUTION ---")

    test_question = "What is the latest stable version of Kubernetes?"
    print(f"\nUser Query: {test_question}\n")

    result = crag_app.invoke(initial_state(test_question))

    print("\n==================================")
    print(f"FINAL GENERATED ANSWER (grounded: {result['grounded']}, checks: {result['retries']}):")
    print("==================================")
    print(result["generation"])