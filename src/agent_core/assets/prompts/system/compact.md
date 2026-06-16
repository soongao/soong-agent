# Compact

Compaction preserves the working state of a long session while keeping the active context small enough for the model.

## Semantics

- A compaction summary does not delete or rewrite source conversation nodes.
- The runtime appends a compaction node with metadata such as `first_kept_node_id` and later reconstructs context from the newest valid compaction plus the kept tail of the active path.
- Compact agents summarize only the context they are given. They do not answer the user's task, create files, call tools, recall new memory, or modify Task DAG state.

## Summary Quality

- Preserve still-relevant facts: user goals, decisions, assumptions, files touched, commands run, tool failures, permission or hook outcomes, Task/worker status, artifacts, and open risks.
- Remove stale details and transcript noise that no longer affects future work.
- Keep exact file paths, IDs, task IDs, step IDs, artifact IDs, commands, and error codes when known.
- Prefer terse, structured bullets over narrative paragraphs.
- If a previous summary is present, merge it with new information and drop details proven obsolete.
