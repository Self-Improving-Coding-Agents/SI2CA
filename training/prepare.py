"""Convert selected policy trajectories to SFT JSONL; drop invalid/overlong examples."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training/backend"))


def normalize(messages):
    clean = []
    for message in messages:
        role = message.get("role")
        if role not in ("system", "user", "assistant", "tool"):
            raise ValueError("invalid role")
        m = {k:v for k,v in message.items() if k in ("role","content","reasoning_content","tool_calls","tool_call_id","name")}
        for call in m.get("tool_calls") or []:
            fn = call.get("function", {})
            if fn.get("name") != "bash":
                raise ValueError("tool must be bash")
            args = fn.get("arguments")
            if isinstance(args,str):
                args=json.loads(args)
            if not isinstance(args,dict) or not isinstance(args.get("command"),str) or not args["command"].strip():
                raise ValueError("invalid bash arguments")
            fn["arguments"] = args
        clean.append(m)
    if not any(m.get("tool_calls") for m in clean):
        raise ValueError("no actions")
    return clean


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--traj-dir", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-length", type=int, default=131072)
    p.add_argument("--task-ids", help="Optional common task-ID file shared by all three arms")
    a=p.parse_args()
    from transformers import AutoTokenizer
    from slime.utils.mask_utils import MultiTurnLossMaskGenerator
    from si2ca.runtime.strategy import BASH_TOOL
    tok=AutoTokenizer.from_pretrained(a.tokenizer, trust_remote_code=True)
    masker=MultiTurnLossMaskGenerator(tok, tokenizer_type="qwen3_5")
    allowed=set(Path(a.task_ids).read_text().splitlines()) if a.task_ids else None
    out=Path(a.out)
    if out.exists():raise FileExistsError(out)
    out.parent.mkdir(parents=True,exist_ok=True)
    rejected=[];retained=[];seen=set()
    with out.open('x') as stream:
        for file in sorted(Path(a.traj_dir).rglob('*.json')):
            try:
                obj=json.loads(file.read_text());result=obj.get('result',obj)
                iid=result.get('instance_id') or obj.get('instance_id')
                if not iid or (allowed is not None and iid not in allowed):continue
                if iid in seen:raise ValueError('duplicate task; select one sample explicitly')
                seen.add(iid)
                if result.get('error') or result.get('abort') or result.get('exit_code',0)!=0:
                    raise ValueError('trajectory did not exit cleanly')
                messages=normalize(obj['messages'])
                ids,mask=masker.get_loss_mask(messages,tools=[BASH_TOOL])
                if len(ids)>a.max_length:raise ValueError('over context limit')
                if len(ids)!=len(mask) or not any(mask):raise ValueError('invalid loss mask')
                stream.write(json.dumps({'messages':messages,'tools':[BASH_TOOL],
                    'metadata':{'instance_id':iid,'reward':result.get('reward',0),'tokens':len(ids),'loss_tokens':sum(mask)}},ensure_ascii=False)+'\n')
                retained.append(iid)
            except (ValueError,KeyError,TypeError) as exc:
                rejected.append({'file':str(file),'reason':str(exc)})
    audit={'retained':len(retained),'retained_ids':retained,'rejected':rejected,
           'missing_requested_ids':sorted(allowed-set(retained)) if allowed is not None else None,
           'recipe':'all clean exits, solved and unsolved; no truncation; qwen3_5 assistant mask'}
    out.with_suffix('.audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    print(f'Retained {len(retained)}, rejected {len(rejected)}. Audit: {out.with_suffix(".audit.json")}')


if __name__=='__main__':main()
