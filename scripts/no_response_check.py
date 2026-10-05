#!/usr/bin/env python3
"""Are the 35 `no_response` rollouts real, or transport failures mislabelled?

Decisive test: the episode's OWN `cat /workdir/response.txt` tool result. If the
model ran cat and the container printed the answer back, the file existed and
was non-empty when the episode ended -- so the scorer's later "response.txt is
empty" verdict is about the scorer's exec, not about the model.
"""
import json, re, sys, glob, collections
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from inspect_ai.log import read_eval_log

# Which run directory to check. Defaults to the kimi_full2 root this script
# was written for, so the original invocation is unchanged; pass a root to
# apply the same check to a later run (e.g. logs/kimi_full3).
ROOT = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "logs/kimi_full2"

SSH = re.compile(r"exit status 255|dial-stdio", re.I)
NITRO = re.compile(r"\bnitrogen\b", re.I)
rows=[]
for f in sorted(glob.glob(f"{ROOT}/**/*.eval", recursive=True)):
    log=read_eval_log(f)
    cond=(log.eval.metadata or {}).get("condition","?")
    for s in (log.samples or []):
        sc=list(s.scores.values())[0] if s.scores else None
        md=(sc.metadata or {}) if sc else {}
        if md.get("outcome")!="no_response": continue
        msgs=s.messages or []
        wrote=False; cat_ok=False; cat_ran=False; leak=False
        for i,m in enumerate(msgs):
            for tc in (getattr(m,"tool_calls",None) or []):
                cmd=str((tc.arguments or {}).get("cmd") or (tc.arguments or {}).get("command") or "")
                if re.search(r">\s*/workdir/response\.txt", cmd): wrote=True
                if re.search(r"\bcat\b[^|]*/workdir/response\.txt", cmd):
                    cat_ran=True
                    # the NEXT tool message is its result
                    for j in range(i+1, min(i+3,len(msgs))):
                        if msgs[j].role=="tool":
                            if NITRO.search(msgs[j].text or ""): cat_ok=True
                            break
            if m.role=="tool" and SSH.search(m.text or ""): leak=True
        rows.append({"disclosure":md.get("disclosure"),"cond":cond,"epoch":s.epoch,
                     "wrote":wrote,"cat_ran":cat_ran,"cat_echoed_answer":cat_ok,
                     "episode_saw_ssh_error":leak,
                     "scorer_notes_mention_ssh":any(SSH.search(str(n)) for n in (md.get("notes") or []))})
Path(f"{ROOT}/no_response_check.json").write_text(json.dumps(rows,indent=1))
n=len(rows)
print(f"no_response rollouts: {n}")
print(f"  wrote to /workdir/response.txt in-episode : {sum(r['wrote'] for r in rows)}/{n}")
print(f"  ran `cat /workdir/response.txt`           : {sum(r['cat_ran'] for r in rows)}/{n}")
print(f"  ...and the container echoed the answer    : {sum(r['cat_echoed_answer'] for r in rows)}/{n}")
print(f"  episode itself saw an ssh/255 error       : {sum(r['episode_saw_ssh_error'] for r in rows)}/{n}")
print(f"  SCORER notes mention the same ssh/255 err : {sum(r['scorer_notes_mention_ssh'] for r in rows)}/{n}")
print()
c=collections.Counter((r['disclosure'],r['cond']) for r in rows)
for k in sorted(c,key=str): print(f"   {k}: {c[k]}")
