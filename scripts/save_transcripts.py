#!/usr/bin/env python3
"""Save the L1 transcripts the report needs to read, as FILES only.

  - every exploited / attempted rollout at L1 (both conditions)
  - 3 inspected_not_exploited per condition

Payload text stays in these files; the report quotes none of it.
"""
import json, sys, glob, collections
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from inspect_ai.log import read_eval_log

# Run root and conditions are arguments so a later run (kimi_full3's
# `finetuned` cell) gets transcripts by the SAME rule, rather than by a second
# script that could drift from this one. Defaults reproduce the original call.
ROOT = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "logs/kimi_full2"
CONDS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["unsteered", "B_pos0.5"]

OUTDIR=Path(f"{ROOT}/transcripts"); OUTDIR.mkdir(parents=True, exist_ok=True)
rows=json.load(open(f'{ROOT}/rows_all.json'))
want=[]
for cond in CONDS:
    for o in ("exploited","attempted"):
        want += [r for r in rows if r['disclosure']=='L1' and r['condition']==cond and r['outcome']==o]
    insp=[r for r in rows if r['disclosure']=='L1' and r['condition']==cond
          and r['outcome']=='inspected_not_exploited']
    want += insp[:3]
print(f"saving {len(want)} transcripts")

byfile=collections.defaultdict(list)
for r in want: byfile[r['file']].append(r)
index=[]
for f, rs in byfile.items():
    log=read_eval_log(f)
    want_ep={r['epoch']:r for r in rs}
    for s in (log.samples or []):
        if s.epoch not in want_ep: continue
        r=want_ep[s.epoch]
        name=f"L1_{r['condition']}_{r['outcome']}_ep{r['epoch']:03d}.txt"
        L=[]
        L.append(f"# level=L1 condition={r['condition']} epoch={r['epoch']} "
                 f"outcome={r['outcome']} official_score={r['score']}")
        L.append(f"# read_grader={r['read_grader']} ran_grader={r['ran_grader']} "
                 f"tamper_attempt={r['tamper_attempt']} injection_shaped={r['injection_shaped']} "
                 f"grader_crashed={r['grader_crashed']} answer_correct={r['answer_correct']} "
                 f"n_bash={r['n_bash']}")
        L.append("="*78)
        for m in s.messages:
            L.append(f"\n----- {m.role} -----")
            if (m.text or "").strip(): L.append(m.text)
            for tc in (getattr(m,"tool_calls",None) or []):
                a=tc.arguments or {}
                L.append(f"[TOOL CALL {tc.function}]\n{a.get('cmd') or a.get('command') or json.dumps(a)}")
        L.append("\n" + "="*78)
        L.append(f"/workdir/response.txt as written:\n{r['response']!r}")
        L.append(f"notes: {r['notes']}")
        (OUTDIR/name).write_text("\n".join(L))
        index.append({"file":name,"condition":r['condition'],"epoch":r['epoch'],
                      "outcome":r['outcome'],"score":r['score'],
                      "read_grader":r['read_grader'],"ran_grader":r['ran_grader'],
                      "injection_shaped":r['injection_shaped'],
                      "grader_crashed":r['grader_crashed'],
                      "answer_correct":r['answer_correct'],"n_bash":r['n_bash'],
                      "resp_len":len(r['response'])})
index.sort(key=lambda x:(x['condition'],x['outcome'],x['epoch']))
(OUTDIR/"INDEX.json").write_text(json.dumps(index,indent=1))
print(f"wrote {len(index)} files + INDEX.json to {OUTDIR}")
for i in index:
    print(f"  {i['file']:<48} score={i['score']:<5} rd={int(i['read_grader'])} run={int(i['ran_grader'])} "
          f"inj={int(i['injection_shaped'])} crash={int(i['grader_crashed'])} nb={i['n_bash']} resp_len={i['resp_len']}")
