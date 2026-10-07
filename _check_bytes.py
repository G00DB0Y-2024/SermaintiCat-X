from pathlib import Path
text = Path('F:/MyProjects/PDFAI/pdf-backen/ai/ai_io.py').read_text(encoding='utf-8')

# Locate plans header line
i = text.find("# plans.json")
print("found at:", i)
# Print the surrounding bytes
print(repr(text[i:i+60]))
print("len:", len(text[i:i+60]))

# Test find with full string
target = "# plans.json (Layer3 计划队列) — 缓存 + 读写"
print(f"target found: {target in text}")

# Char-by-char
print("--- char check ---")
for c, t in zip(text[i:i+45], target):
    if c != t:
        print(f"diff: file={c!r} {hex(ord(c))}  target={t!r} {hex(ord(t))}")
        break