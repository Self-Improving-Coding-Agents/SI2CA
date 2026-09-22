# Prompt assets

This directory owns SI2CA's built-in model-facing text. Python assembles messages and fills parameters; it does not embed the prompt bodies.

| Files | Used by |
| --- | --- |
| `agent_system.md`, `agent_instance.md` | Coding-agent system and task instructions (Standard, SJ, SL and the Discovered Early-Commit Window Strategy) |
| `agent_observation.md`, `agent_format_error.md` | Tool results and tool-call repair messages |
| `agent_task.md` | Shell-task instructions |
| `agent_readback.md` | Historical edit-readback template retained for reference |
| `bash_tool.json` | Shared bash tool schema, including descriptions, for generation and likelihood scoring |
| `judge_system.md` | SJ without the gold-patch rubric |
| `judge_oracle.md` | Gold-guided SJ in the unified runner |
| `judge_branch_oracle.md`, `judge_action_only.md`, `judge_instance_rubric.md` | Historical rubric variants retained for reference |
| `rubrics.json` | Historical action-family rubric definitions and weights retained for reference |
| `gold_guided_scoring.md`, `gold_guided_scoring_head.md`, `gold_guided_scoring_hint.md` | SL privileged context: tail gold, head gold and hints |
| `hint_generation.md` | Generate hints from an issue and its reference patch |
| `messages.json` | Shared judge/history sections, family guidance, retry messages and shim probe text |

Edit these files in a source checkout, then start a new process; use an editable install when developing. Wheels and source distributions include the assets, so installed runs do not need the checkout. The default agent templates are copied from the pinned mini-swe-agent 2.4.2 configuration and loaded here, not from its installed prompt files. Its non-prompt runtime settings are still read as before.

Keep the existing placeholders and rendering conventions:

- Agent templates use Jinja (`{{task}}`, `{{output.output}}`, `{% ... %}`).
- Judge templates use Python formatting (`{formula}`, `{score_schema}`); literal JSON braces are doubled. Most parameterized entries in `messages.json` use the same convention; its strategy-observation fallback uses Jinja and its hint retry uses `%` formatting.
- SL and hint-generation templates use literal replacement of `{{reference_patch}}`, `{{problem_statement}}` and `{{gold_patch}}`. Inserted code and patches are never reinterpreted as templates.
- `agent_readback.md` retains its four `%s` substitutions. Leading whitespace and prompt boundaries are intentional; loaders remove only the file-ending newline where the original embedded template had none. SL's existing leading-comment and outer-whitespace handling is unchanged.

The `SWE_CC_PROMPT` and strategy-runner `--inject-template` overrides remain supported. The removed legacy drivers' `MINIMAL_SYSTEM_TEMPLATE_FILE` and `MINIMAL_INSTANCE_TEMPLATE_FILE` overrides are no longer used. Tokenizer chat templates still belong to the selected model. Task statements, reference patches and per-issue rubrics are dataset inputs, not built-in prompts. Recursive-search [proposer and recorder skills](../../skills/) live directly in `skills/*.md`; they are not duplicated here.

Run `python -m unittest discover -s tests -v` after changes. Prompt regression tests record the original rendered text; update those expectations deliberately when changing an experiment's wording.
