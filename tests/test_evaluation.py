from evaluation.tool_selection_eval import run_evaluation

def test_evaluation_harness():
    # Will fail if run_evaluation calls sys.exit(1)
    run_evaluation()
