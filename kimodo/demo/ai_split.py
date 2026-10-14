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


def rule_split(text):
    parts = [p.strip(" ，,。") for p in SPLIT_RE.split(text) if p.strip(" ，,。")]
    if len(parts) <= 1:
        parts = [text.strip()]
    return parts


def glm_split(text):
    body = json.dumps({
        "model": GLM_MODEL,
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
        "https://open.bigmodel.cn/api/paas/v4/chat/completions",
        data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {GLM_KEY}"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
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
    when GLM is configured, otherwise the original Chinese segments)."""
    if GLM_KEY:
        try:
            poses = glm_split(text)
            if poses and len(poses) >= 2:
                return poses
        except Exception:
            pass
    return rule_split(text)
