# analysis_update Agent Notes

Use manuscript style in Methods and Results.
Avoid bullet lists and avoid inline code formatting for variables, covariates, and model terms.
Write variables and model symbols as plain text or mathematical notation instead.

Use LaTeX math delimiters consistently.
Use `$...$` for inline formulas and `$$...$$` for display formulas.
Prefer inline formulas when expressions are short.
Use display formulas only when the full model form is needed for readability.

Keep line wrapping git-friendly.
Use single line breaks to keep diffs readable, without forcing a new rendered paragraph after every sentence.

If quality-control filters are used for an analysis, state them explicitly in that analysis method subsection.

In results text, report both direction and magnitude for key effects, with corresponding p and q values when available.

For figure references in Results, provide both a minimal figure set and an optional figure set using inline links.

Deletion markup conventions used by the collaborator:
Inline deletion requests may be marked by italic text with single asterisks.
Block deletion requests may be marked as block quotes.
