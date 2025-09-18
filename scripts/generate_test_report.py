import json
import os
import base64
from jinja2 import Environment, FileSystemLoader
from datetime import datetime


def main():
    """
    Generates a test report by rendering a Jinja2 template with data from environment variables.
    """

    # --- Data Loading ---

    # General Info
    title = "Test Report"
    action_url = os.environ.get("GITHUB_ACTION_URL", "")

    # Branch Info
    branch_success = os.environ.get("BRANCH_RESULT", "success") == "success"
    branch_actual = os.environ.get("BRANCH_ACTUAL", "unknown")
    branch_target = os.environ.get("BRANCH_TARGET", "dev")

    # Changed Files Info
    files_success = os.environ.get("FILES_RESULT", "success") == "success"
    forbidden_changes_b64 = os.environ.get("FORBIDDEN_CHANGES_B64", "")
    not_allowed_files = []
    if forbidden_changes_b64:
        not_allowed_files = (
            base64.b64decode(forbidden_changes_b64).decode().strip().split("\n")
        )

    # Framework Tests Info
    test_outcomes_str = os.environ.get("TEST_OUTCOMES", "{}")
    test_outcomes = json.loads(test_outcomes_str)
    tests_success = all(outcome == "success" for outcome in test_outcomes.values())
    test_results = {
        game: outcome == "success" for game, outcome in test_outcomes.items()
    }

    # Test Logs
    test_logs = {}

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    reports_dir = os.path.join(repo_root, "reports")
    if os.path.isdir(reports_dir):
        for fname in os.listdir(reports_dir):
            if not (fname.startswith("pytest_output_") and fname.endswith(".txt")):
                continue
            game = fname[len("pytest_output_") : -len(".txt")]
            fpath = os.path.join(reports_dir, fname)
            try:
                with open(fpath, "r", encoding="utf-8", errors="replace") as rf:
                    test_logs[game] = rf.read()
            except Exception as e:
                # Keep going; file might be unreadable
                test_logs[game] = f"<error reading {fname}: {e}>"

    # --- Template Rendering ---

    # Prepare context for Jinja2
    result = {
        "title": title,
        "action_url": action_url,
        "branch": {
            "success": branch_success,
            "actual": branch_actual,
            "target": branch_target,
        },
        "files": {
            "success": files_success,
            "not_allowed": not_allowed_files,
        },
        "tests": {
            "success": tests_success,
            "logs": test_logs,
            "results": test_results,
        },
        "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"),
    }

    # Setup Jinja2 environment
    template_dir = os.path.join(os.path.dirname(__file__), "..", "tests", "templates")
    env = Environment(
        loader=FileSystemLoader(template_dir), trim_blocks=True, lstrip_blocks=True
    )
    template = env.get_template("pr_comment.md.j2")

    # Render the template
    output = template.render(result=result)

    # Write the output to a file
    with open("ci-report.md", "w") as f:
        f.write(output)

    print("✅ Generated CI report successfully to ci-report.md")


if __name__ == "__main__":
    main()
