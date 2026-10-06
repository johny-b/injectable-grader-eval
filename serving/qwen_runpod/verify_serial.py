import json, sys
sys.path.insert(0, "/work/workspace/qwen_serve")
import importlib.util
spec = importlib.util.spec_from_file_location("v", "/work/workspace/qwen_serve/verify_server.py")
src = open("/work/workspace/qwen_serve/verify_server.py").read().split("out = {}")[0]
exec(src)
prev = json.load(open("/work/workspace/qwen_serve/verify_server.json"))
a, b = prev["base_a"][1]["reasoning"], prev["base_b"][1]["reasoning"]
i = next((k for k, (x, y) in enumerate(zip(a, b)) if x != y), None)
print("batched base_a vs base_b first diff at char", i, "of", len(a), ":", repr(a[max(0,i-60):i+40]), "||", repr(b[max(0,i-60):i+40]))
res = {}
for k in ["s_base1", "s_base2", "s_base3", "s_steer1", "s_steer2"]:
    res[k] = chat(("0007", 0.3) if "steer" in k else None, 0.0, 600)
t = lambda k: (res[k][1]["reasoning"], res[k][1]["content"])
print("serial T=0 base x3 identical:", t("s_base1") == t("s_base2") == t("s_base3"))
print("serial T=0 steer0.3 x2 identical:", t("s_steer1") == t("s_steer2"))
print("serial steer0.3 != base:", t("s_steer1") != t("s_base1"))
a, b = t("s_base1")[0], t("s_steer1")[0]
i = next((k for k, (x, y) in enumerate(zip(a, b)) if x != y), None)
print("base vs steer first diff char", i, ":", repr(a[i-40:i+80]), "\n  ||", repr(b[i-40:i+80]))
json.dump(res, open("/work/workspace/qwen_serve/verify_serial.json", "w"), indent=1)
