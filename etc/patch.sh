for M in Qwen2.5-VL-72B-Instruct-AWQ Qwen2.5-VL-32B-Instruct-AWQ QwQ-32B; do
  f=/mnt/data/models/$M/tokenizer_config.json
  cp "$f" "$f.bak"
  python3 - "$f" <<'PY'
import json,sys
p=sys.argv[1]; d=json.load(open(p))
d["tokenizer_class"]="Qwen2TokenizerFast"
json.dump(d,open(p,"w"),ensure_ascii=False,indent=2)
print("patched",p)
PY
done
