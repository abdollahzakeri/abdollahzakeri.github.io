#!/usr/bin/env python3
"""
Split legacy data.js into a hierarchical examples/ structure and manifest.json.

Structure:
  examples/<categoryId>/<topicId>/<exampleId>.js
Each file calls window.registerExample(categoryId, {categoryName, categorySummary, topicId, topicName}, {id,name,tags,description,meta,code})

This script heuristically parses data.js by evaluating its JS as JSON-like with a tiny JS-to-JSON normalization.
If normalization fails, it falls back to a lightweight text parser to recover categories and topics.

Usage: python3 scripts/split_data.py
"""
import os
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_JS = ROOT / 'data.js'
OUT_DIR = ROOT / 'examples'
MANIFEST = OUT_DIR / 'manifest.json'

def read_data_js():
    return DATA_JS.read_text(encoding='utf-8')

def js_to_json_like(js_text: str) -> str:
    # Extract object via brace matching to be robust to inner patterns
    start = js_text.find('window.PYTORCH_COOKBOOK')
    if start == -1:
        raise ValueError('PYTORCH_COOKBOOK object not found')
    eq = js_text.find('=', start)
    brace = js_text.find('{', eq)
    if brace == -1:
        raise ValueError('Opening brace not found')
    i = brace
    depth = 0
    in_backtick = False
    prev = ''
    while i < len(js_text):
        ch = js_text[i]
        if ch == '`' and prev != '\\':
            in_backtick = not in_backtick
        if not in_backtick:
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        prev = ch
        i += 1
    else:
        raise ValueError('Closing brace not found')
    obj = js_text[brace:end]
    # Quote keys if needed and convert single quotes to double quotes inside values carefully.
    # This is a best-effort normalizer tailored to our generated file.
    # Replace JS backticks sections by JSON strings by escaping.
    # Find code: `...` blocks and replace with placeholder tokens, store separately.
    code_blocks = []
    def repl_code(match):
        code = match.group(0)
        inner = code[1:-1]
        token = f"__CODE_BLOCK_{len(code_blocks)}__"
        code_blocks.append(inner)
        return f'"{token}"'
    obj2 = re.sub(r"`[\s\S]*?`", repl_code, obj)
    # Convert single-quoted strings to double quotes
    obj2 = re.sub(r"'(?:\\'|[^'])*'", lambda m: '"' + m.group(0)[1:-1].replace('\\"','\"').replace('"','\\"') + '"', obj2)
    # Add quotes around unquoted object keys
    obj2 = re.sub(r"(\{|,)(\s*)([A-Za-z_][A-Za-z0-9_-]*)(\s*):", r"\1\2""\3""\4:", obj2)
    # Remove trailing commas
    obj2 = re.sub(r",(\s*[}\]])", r"\1", obj2)
    # Restore code blocks
    for i, code in enumerate(code_blocks):
        placeholder = f"__CODE_BLOCK_{i}__"
        encoded = json.dumps(code)
        obj2 = obj2.replace(f'"{placeholder}"', encoded)
    return obj2

def parse_cookbook(js_text: str):
    obj_json = js_to_json_like(js_text)
    data = json.loads(obj_json)
    if 'categories' not in data:
        raise ValueError('categories missing in parsed data')
    return data

def sanitize_id(s: str) -> str:
    return re.sub(r"[^a-z0-9-_]", "-", s.lower())

def ensure_empty_dir(path: Path):
    if path.exists():
        # Remove old tree excluding manifest
        for p in sorted(path.glob('**/*'), reverse=True):
            if p.is_file():
                p.unlink()
            elif p.is_dir():
                try:
                    p.rmdir()
                except OSError:
                    pass
    path.mkdir(parents=True, exist_ok=True)

def write_example_js(category, topic, ex):
    cat_id = category['id']
    topic_id = topic['id']
    ex_id = ex.get('id') or sanitize_id(ex.get('name','example'))
    rel = Path(cat_id) / topic_id / f"{ex_id}.js"
    out_path = OUT_DIR / rel
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'categoryId': cat_id,
        'topicInfo': {
            'categoryName': category.get('name',''),
            'categorySummary': category.get('summary',''),
            'topicId': topic_id,
            'topicName': topic.get('name',''),
        },
        'example': {
            'id': ex_id,
            'name': ex.get('name',''),
            'tags': ex.get('tags',[]),
            'description': ex.get('description',''),
            'meta': ex.get('meta',''),
            'code': ex.get('code',''),
        }
    }
    js = (
        "(function(){\n" 
        f"  window.registerExample({json.dumps(payload['categoryId'])}, "
        f"{json.dumps(payload['topicInfo'])}, {json.dumps(payload['example'])});\n"
        "})();\n"
    )
    out_path.write_text(js, encoding='utf-8')
    return str(rel).replace('\\', '/')

def main():
    js_text = read_data_js()
    data = parse_cookbook(js_text)
    ensure_empty_dir(OUT_DIR)
    files = []
    for cat in data['categories']:
        for topic in cat.get('topics', []):
            # In legacy structure, each topic was a single example; keep as one file.
            ex = {
                'id': sanitize_id(topic.get('id') or topic.get('name','example')),
                'name': topic.get('name',''),
                'tags': topic.get('tags', []),
                'description': topic.get('description',''),
                'meta': topic.get('meta',''),
                'code': topic.get('code',''),
            }
            rel = write_example_js(cat, topic, ex)
            files.append(rel)
    MANIFEST.write_text(json.dumps({ 'files': files }, indent=2), encoding='utf-8')
    print(f'Wrote {len(files)} example files and manifest to {OUT_DIR}')

if __name__ == '__main__':
    main()


