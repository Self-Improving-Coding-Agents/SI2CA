"""Extract reference patches without confusing them with tests or arbitrary answers."""
import re

FIELDS = ("metadata.gold_patch", "gold_patch", "metadata.reference_patch", "reference_patch",
          "metadata.golden_patch", "golden_patch", "metadata.reference_solution", "reference_solution",
          "metadata.patch", "patch", "metadata.solution.patch", "solution.patch",
          "metadata.answer", "answer", "metadata.solution", "solution")


def field_value(row, field):
    value = row
    for key in field.split('.'):
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def extract_patch(row, field=None):
    found = {}
    for name in (field,) if field else FIELDS:
        value = field_value(row, name)
        if not isinstance(value, str) or not value.strip():
            continue
        fences = re.findall(r"```(?:diff|patch)\s*\n(.*?)```", value, re.S)
        if fences:
            value = '\n'.join(fences)
        # Hunk context prefixes and trailing whitespace are part of the diff.
        # In particular, dropping the final newline can make git reject it.
        value = value.lstrip('\r\n')
        if not value.endswith('\n'):
            value += '\n'
        # Explicit patch-named fields are authoritative. Generic answer/solution
        # fields must contain a unified diff, not prose or implementation code.
        generic = name.split('.')[-1] in ('answer', 'solution', 'reference_solution')
        if generic and not (re.search(r'^diff --git ', value, re.M) or
                            (re.search(r'^--- ', value, re.M) and re.search(r'^\+\+\+ ', value, re.M))):
            continue
        found[name] = value
    if len(set(found.values())) > 1:
        iid = row.get('metadata', {}).get('instance_id', row.get('instance_id', '?'))
        raise ValueError(f"{iid}: conflicting reference patches in {', '.join(found)}; select --gold-field explicitly")
    return next(iter(found.values()), None)
