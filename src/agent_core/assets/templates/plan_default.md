# Default Plan Template

Create a decision-complete Markdown plan for the requested goal. The plan should be specific enough that another agent can implement it without choosing architecture, APIs, test scope, or sequencing.

Use this structure:

1. `Goal`: one sentence describing the verified end state.
2. `Scope`: concrete behavior, modules, files, commands, or user workflows that are in scope.
3. `Approach`: the chosen implementation strategy and the alternatives deliberately rejected.
4. `Interfaces`: public APIs, schemas, tool names, CLI flags, config keys, files, or data formats that will change.
5. `Steps`: ordered implementation steps with observable outcomes.
6. `Edge Cases`: important failure modes, permissions, rollback behavior, compatibility, and data migration concerns.
7. `Verification`: exact tests, commands, or manual checks that prove completion.
8. `Assumptions`: unresolved facts or user preferences that the implementer must preserve.

Keep the plan concise but complete. Use the suggested project plan directory when the caller provides one. Do not write implementation code in the plan.
