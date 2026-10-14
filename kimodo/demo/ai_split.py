"""Split a natural-language action description into sequential key poses.

Rule-based: split on Chinese sequential connectors (然后/接着/再/最后/之后/随后).
Optional GLM enhancement via GLM_API_KEY env (outputs English pose prompts).
"""
import json
import os
import re
import urllib.request

SPLIT_RE = re.compile(r"然后|接着|再|最后|之后|随后")

GLM_KEY = os.environ.get("GLM_API_KEY", "")
GLM_MODEL = os.environ.get("GLM_MODEL", "glm-4-flash")
AITRADE_CONFIG = "/home/applo/project/AItrade/aitrade/config.py"


def _read_llm_cfg():
    """GLM_API_KEY env first, then the AItrade config's OpenAI-compatible LLM."""
    if GLM_KEY:
        base = os.environ.get("GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")
        return base.rstrip("/"), GLM_KEY, GLM_MODEL
    try:
        with open(AITRADE_CONFIG, encoding="utf-8") as f:
            src = f.read()

        def val(name):
            m = re.search(rf"^{name}\s*=\s*['\"]([^'\"]+)['\"]", src, re.M)
            return m.group(1) if m else ""

        base, key, model = val("LLM_BASE_URL"), val("LLM_API_KEY"), val("LLM_MODEL")
        if base and key:
            return base.rstrip("/"), key, model
    except Exception:
        pass
    return None, None, GLM_MODEL


def rule_split(text):
    parts = [p.strip(" ，,。") for p in SPLIT_RE.split(text) if p.strip(" ，,。")]
    if len(parts) <= 1:
        parts = [text.strip()]
    return parts


def glm_split(text):
    base, key, model = _read_llm_cfg()
    if not key:
        return None
    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": (
                "把用户的动作描述拆分成 2-5 个连续的关键姿势。只输出 JSON 数组，"
                '格式：["英文姿势描述1", "英文姿势描述2", ...]，描述要具体可视化，'
                "供动作生成模型使用。")},
            {"role": "user", "content": text},
        ],
        "temperature": 0.2,
    }).encode("utf-8")
    req = urllib.request.Request(
        base + "/chat/completions",
        data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {key}"},
    )
    with urllib.request.urlopen(req, timeout=90) as r:
        resp = json.loads(r.read().decode("utf-8"))
    content = resp["choices"][0]["message"]["content"]
    m = re.search(r"\[.*\]", content, re.S)
    if not m:
        return None
    arr = json.loads(m.group(0))
    poses = [str(p).strip() for p in arr if str(p).strip()]
    return poses or None


def split_action_text(text):
    """Return a list of >=1 sequential key-pose descriptions (English preferred
    when an LLM endpoint is configured, otherwise the original Chinese segments)."""
    try:
        poses = glm_split(text)
        if poses and len(poses) >= 2:
            return poses
    except Exception:
        pass
    return rule_split(text)
