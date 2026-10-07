import re
from pathlib import Path
text = Path('F:/MyProjects/PDFAI/pdf-backen/ai/ai_io.py').read_text(encoding='utf-8')
# Find lambda / plans block markers
markers = [
    "outreach_lambda",
    "plans.json (Layer3",
    "Layer2 泊松速率",
    "_load_lambda",
    "_clamp_lambda",
    "_save_lambda",
    "PLANS_FILE",
    "PLANS_KEEP_MAX",
    "_load_plans",
    "_save_plans",
    "_plans_cache",
    "_plans_lock",
    "_lambda_cache",
    "_lambda_clamp_hits",
    "OUTREACH_LAMBDA",
    "from .ai_outreach",
]
for m in markers:
    cnt = text.count(m)
    print(f"  {m}: {cnt}")

# Output sections for visual check
print("---")
print("LAMBDA_START?", "# outreach_lambda" in text)
print("PLANS_START?", "# plans.json (Layer3" in text)
print("TRACK_START?", "# Crystal_track (跨论文" in text)