from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .io import read_jsonl
from .package import build_prompt_package
from .search import search_index
from .status import build_status_report


ROOT = Path(__file__).resolve().parents[2]
INDEX = ROOT / "data" / "index" / "quotes_index.json"
TAGGED = ROOT / "data" / "processed" / "quotes_tagged.jsonl"
RAW_DIR = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed" / "quotes_clean.jsonl"
DISTILL_DIR = ROOT / "data" / "distill"
PERSONA_DIR = ROOT / "persona"
CLIPBOARD = ROOT / "clipboard.md"
PACKAGE_TEMPLATE = ROOT / "prompts" / "package_template.md"
HISTORY_PATH = ROOT / "data" / "web_history.json"


class GenerateRequest(BaseModel):
    question: str = Field(min_length=1)
    context: str = ""
    persona: str = "meta_thinking.md"
    domains: list[str] = Field(default_factory=list)
    quote_types: list[str] = Field(default_factory=list)
    time_sensitivities: list[str] = Field(default_factory=list)
    top_k: int = 8
    max_tokens: int = 3500


def create_app() -> FastAPI:
    app = FastAPI(title="HYMY RAG UI")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _html()

    @app.get("/api/meta")
    def meta() -> dict[str, Any]:
        tagged_rows = read_jsonl(TAGGED)
        domain_counts: Counter[str] = Counter()
        type_counts: Counter[str] = Counter()
        time_counts: Counter[str] = Counter()
        for row in tagged_rows:
            for domain in row.get("domains", []):
                domain_counts[str(domain)] += 1
            type_counts[str(row.get("type", "unknown"))] += 1
            time_counts[str(row.get("time_sensitivity", "unknown"))] += 1
        return {
            "domains": sorted(domain_counts),
            "quote_types": sorted(type_counts),
            "time_sensitivities": sorted(time_counts),
            "domain_counts": dict(domain_counts),
            "type_counts": dict(type_counts),
            "time_counts": dict(time_counts),
            "status": build_status_report(
                raw_dir=RAW_DIR,
                processed_path=PROCESSED,
                tagged_path=TAGGED,
                index_path=INDEX,
                persona_dir=PERSONA_DIR,
                distill_dir=DISTILL_DIR,
            ),
            "history": _load_history(),
        }

    @app.post("/api/generate")
    def generate(request: GenerateRequest) -> dict[str, Any]:
        if not INDEX.exists():
            raise HTTPException(status_code=400, detail="本地索引不存在，请先运行 build-index")
        if not PACKAGE_TEMPLATE.exists():
            raise HTTPException(status_code=400, detail="package 模板不存在，请检查 prompts/package_template.md")
        normalized_top_k = min(max(request.top_k, 5), 20)
        preferred_time = request.time_sensitivities[0] if len(request.time_sensitivities) == 1 else None
        results = search_index(
            INDEX,
            request.question,
            top_k=normalized_top_k,
            domains=request.domains,
            quote_types=request.quote_types,
            time_sensitivities=request.time_sensitivities,
            preferred_time_sensitivity=preferred_time,
        )
        package = build_prompt_package(
            question=request.question,
            results=results,
            template_path=PACKAGE_TEMPLATE,
            persona_dir=PERSONA_DIR,
            output_path=CLIPBOARD,
            current_context=request.context,
            persona_name=request.persona,
            max_tokens=request.max_tokens,
        )
        token_estimate = max(1, len(package) // 2)
        retrieved = [
            {
                "source_id": row.get("source_id", ""),
                "date": row.get("date", "unknown"),
                "type": row.get("type", "unknown"),
                "time_sensitivity": row.get("time_sensitivity", "unknown"),
                "score": row.get("score", 0),
                "summary": row.get("one_line_summary", ""),
                "content": row.get("content", ""),
            }
            for row in results
        ]
        snapshot = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "question": request.question,
            "context": request.context,
            "persona": request.persona,
            "top_k": normalized_top_k,
            "token_estimate": token_estimate,
            "domains": request.domains,
            "quote_types": request.quote_types,
            "time_sensitivities": request.time_sensitivities,
        }
        history = _save_history(snapshot)
        return {
            "markdown": package,
            "token_estimate": token_estimate,
            "retrieved": retrieved,
            "history": history,
            "clipboard_path": str(CLIPBOARD),
        }

    return app


def _load_history() -> list[dict[str, Any]]:
    if not HISTORY_PATH.exists():
        return []
    try:
        data = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def _save_history(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    history = _load_history()
    history.insert(0, snapshot)
    history = history[:20]
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    HISTORY_PATH.write_text(json.dumps(history, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return history


def _html() -> str:
    return """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>HYMY RAG Console</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,500;6..72,700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
  <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
  <style>
    :root {
      --bg: #111418;
      --panel: #171d22;
      --panel-2: #1d252c;
      --line: rgba(255,255,255,0.09);
      --text: #eaf0f1;
      --muted: #93a3ab;
      --accent: #d8a96b;
      --accent-2: #6dc3b1;
      --danger: #c67777;
      --shadow: 0 24px 64px rgba(0, 0, 0, 0.28);
      --radius: 18px;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "IBM Plex Sans", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(216, 169, 107, 0.10), transparent 22rem),
        radial-gradient(circle at bottom right, rgba(109, 195, 177, 0.08), transparent 24rem),
        var(--bg);
      color: var(--text);
    }
    button, input, textarea, select { font: inherit; }
    .shell {
      min-height: 100vh;
      padding: 28px;
    }
    .frame {
      max-width: 1480px;
      margin: 0 auto;
      display: grid;
      gap: 20px;
    }
    .mast {
      display: grid;
      grid-template-columns: 1.3fr 0.7fr;
      gap: 18px;
      align-items: end;
    }
    .titleblock {
      background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01));
      border: 1px solid var(--line);
      border-radius: var(--radius);
      padding: 24px 26px 22px;
      box-shadow: var(--shadow);
    }
    .eyebrow {
      color: var(--accent);
      font-size: 12px;
      letter-spacing: 0.16em;
      text-transform: uppercase;
      margin-bottom: 10px;
    }
    .title {
      margin: 0;
      font-family: "Newsreader", serif;
      font-size: clamp(36px, 5vw, 58px);
      line-height: 0.96;
      font-weight: 700;
    }
    .sub {
      margin-top: 12px;
      color: var(--muted);
      max-width: 54rem;
      line-height: 1.6;
    }
    .summaryband {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
    }
    .metric {
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 16px;
      background: rgba(255,255,255,0.02);
    }
    .metric .label {
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }
    .metric .value {
      margin-top: 10px;
      font-size: 28px;
      font-weight: 600;
    }
    .workspace {
      display: grid;
      grid-template-columns: minmax(340px, 0.9fr) minmax(460px, 1.1fr);
      gap: 20px;
    }
    .panel {
      background: linear-gradient(180deg, rgba(255,255,255,0.025), rgba(255,255,255,0.01));
      border: 1px solid var(--line);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
    }
    .controls {
      padding: 20px;
      display: grid;
      gap: 18px;
    }
    .sectiontitle {
      font-size: 13px;
      color: var(--muted);
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }
    .field {
      display: grid;
      gap: 8px;
    }
    .field label {
      color: var(--muted);
      font-size: 13px;
    }
    .field input, .field textarea, .field select {
      width: 100%;
      border-radius: 14px;
      border: 1px solid rgba(255,255,255,0.08);
      background: rgba(8, 12, 15, 0.62);
      color: var(--text);
      padding: 14px 15px;
      outline: none;
      transition: border-color 160ms ease, transform 160ms ease, background 160ms ease;
    }
    .field textarea {
      min-height: 112px;
      resize: vertical;
    }
    .field input:focus, .field textarea:focus, .field select:focus {
      border-color: rgba(216, 169, 107, 0.7);
      background: rgba(10, 14, 18, 0.82);
    }
    .checkgrid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(110px, 1fr));
      gap: 10px;
    }
    .checkchip {
      border: 1px solid rgba(255,255,255,0.09);
      border-radius: 14px;
      padding: 10px 12px;
      background: rgba(255,255,255,0.02);
      display: flex;
      align-items: center;
      gap: 8px;
      min-height: 48px;
    }
    .checkchip input {
      width: 16px;
      height: 16px;
      accent-color: var(--accent);
      margin: 0;
    }
    .sliderrow {
      display: grid;
      gap: 12px;
    }
    .sliderbar {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 14px;
      align-items: center;
    }
    input[type="range"] {
      accent-color: var(--accent-2);
    }
    .btnrow {
      display: flex;
      gap: 12px;
      align-items: center;
      flex-wrap: wrap;
    }
    .primary {
      border: 0;
      border-radius: 14px;
      background: linear-gradient(135deg, var(--accent), #f1c88c);
      color: #1a1713;
      padding: 14px 18px;
      font-weight: 700;
      cursor: pointer;
    }
    .secondary {
      border: 1px solid rgba(255,255,255,0.12);
      border-radius: 14px;
      background: rgba(255,255,255,0.02);
      color: var(--text);
      padding: 14px 18px;
      cursor: pointer;
    }
    .pillnote {
      color: var(--muted);
      font-size: 13px;
    }
    .results {
      display: grid;
      grid-template-rows: auto 1fr;
      min-height: 760px;
    }
    .resulthead {
      padding: 18px 20px;
      border-bottom: 1px solid var(--line);
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      flex-wrap: wrap;
    }
    .tokenbox {
      display: flex;
      gap: 10px;
      align-items: center;
      color: var(--muted);
      font-size: 13px;
    }
    .tokenbox strong {
      color: var(--accent);
      font-size: 18px;
    }
    .render {
      padding: 24px 24px 28px;
      overflow: auto;
    }
    .render article {
      max-width: 66rem;
      margin: 0 auto;
      line-height: 1.75;
      color: #dde5e8;
    }
    .render h1, .render h2, .render h3 {
      font-family: "Newsreader", serif;
      font-weight: 700;
      line-height: 1.1;
      margin: 1.4em 0 0.6em;
      color: #f4f0ea;
    }
    .render pre, .render code {
      font-family: "IBM Plex Mono", monospace;
    }
    .render pre {
      background: rgba(0,0,0,0.28);
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: 14px;
      padding: 14px;
      overflow: auto;
    }
    .render blockquote {
      margin: 16px 0;
      padding-left: 14px;
      border-left: 2px solid rgba(216, 169, 107, 0.8);
      color: #c9d4d8;
    }
    .lower {
      display: grid;
      grid-template-columns: 1.1fr 0.9fr;
      gap: 20px;
    }
    .stack {
      display: grid;
      gap: 20px;
    }
    .retrieved, .history, .statuspanel {
      padding: 20px;
    }
    .quotelist {
      display: grid;
      gap: 12px;
      margin-top: 14px;
    }
    .quoteitem {
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: 14px;
      padding: 14px 15px;
      background: rgba(255,255,255,0.02);
    }
    .quotehead {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      flex-wrap: wrap;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 8px;
    }
    .quotecontent {
      line-height: 1.6;
      color: #e6ecee;
    }
    .quotesummary {
      margin-top: 8px;
      font-size: 13px;
      color: #aac4c1;
    }
    .historylist {
      display: grid;
      gap: 10px;
      margin-top: 14px;
    }
    .historybtn {
      text-align: left;
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: 14px;
      background: rgba(255,255,255,0.02);
      color: var(--text);
      padding: 12px 14px;
      cursor: pointer;
    }
    .historymeta {
      margin-top: 6px;
      color: var(--muted);
      font-size: 12px;
    }
    .dashboard {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 14px;
      margin-top: 14px;
    }
    .donut {
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: 16px;
      padding: 14px;
      background: rgba(255,255,255,0.02);
      min-height: 220px;
      display: grid;
      align-content: start;
      gap: 12px;
    }
    .ring {
      width: 108px;
      aspect-ratio: 1;
      border-radius: 50%;
      margin: 0 auto;
      position: relative;
      background: conic-gradient(var(--accent) 0 40%, var(--accent-2) 40% 72%, #6c7f95 72% 100%);
    }
    .ring::after {
      content: "";
      position: absolute;
      inset: 16px;
      border-radius: 50%;
      background: var(--panel);
      box-shadow: inset 0 0 0 1px rgba(255,255,255,0.06);
    }
    .legend {
      display: grid;
      gap: 8px;
      font-size: 12px;
    }
    .legenditem {
      display: grid;
      grid-template-columns: 10px 1fr auto;
      gap: 8px;
      align-items: center;
      color: var(--muted);
    }
    .swatch {
      width: 10px;
      height: 10px;
      border-radius: 999px;
    }
    .statuspre {
      white-space: pre-wrap;
      font-family: "IBM Plex Mono", monospace;
      line-height: 1.7;
      color: #d2dde1;
      margin: 12px 0 0;
    }
    .empty {
      color: var(--muted);
      padding: 18px 0 6px;
    }
    .danger {
      color: var(--danger);
    }
    @media (max-width: 1100px) {
      .mast, .workspace, .lower { grid-template-columns: 1fr; }
      .dashboard { grid-template-columns: 1fr; }
      .summaryband { grid-template-columns: 1fr; }
      .results { min-height: 560px; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <div class="frame">
      <section class="mast">
        <div class="titleblock">
          <div class="eyebrow">HYMY Local Console</div>
          <h1 class="title">把检索、Persona 和提问包放进同一个工作台。</h1>
          <div class="sub">这个页面不替你思考，它只负责把你现在的问题、上下文和资料库里最该用的那部分，干净地拼成一份能直接拿去问模型的包。</div>
        </div>
        <div class="summaryband">
          <div class="metric"><div class="label">Clipboard</div><div class="value">`clipboard.md`</div></div>
          <div class="metric"><div class="label">Port</div><div class="value">8765</div></div>
          <div class="metric"><div class="label">History</div><div class="value" id="history-count">0</div></div>
        </div>
      </section>

      <section class="workspace">
        <form class="panel controls" id="generator-form">
          <div class="sectiontitle">输入与过滤</div>
          <div class="field">
            <label for="question">用户问题</label>
            <textarea id="question" placeholder="比如：我现在应该把重心放在搞钱还是先修身体和节律？" required></textarea>
          </div>
          <div class="field">
            <label for="context">当前处境 / 上下文</label>
            <textarea id="context" placeholder="可空。比如：2026 年 4 月，我刚离职，现金流只能撑 6 个月。"></textarea>
          </div>
          <div class="field">
            <label for="persona-file">Persona 文件</label>
            <input id="persona-file" value="meta_thinking.md" placeholder="默认 meta_thinking.md">
          </div>
          <div class="field">
            <label>domain 过滤</label>
            <div class="checkgrid" id="domain-filters"></div>
          </div>
          <div class="field">
            <label>type 过滤</label>
            <div class="checkgrid" id="type-filters"></div>
          </div>
          <div class="field">
            <label>time_sensitivity 过滤</label>
            <div class="checkgrid" id="time-filters"></div>
          </div>
          <div class="sliderrow">
            <div class="sectiontitle">检索规模</div>
            <div class="sliderbar">
              <input id="top-k" type="range" min="5" max="20" step="1" value="8">
              <div id="top-k-value">8</div>
            </div>
          </div>
          <div class="btnrow">
            <button class="primary" type="submit" id="generate-btn">生成提问包</button>
            <button class="secondary" type="button" id="clear-btn">清空</button>
            <span class="pillnote" id="hint">默认会同步写入 `clipboard.md`。</span>
          </div>
        </form>

        <section class="panel results">
          <div class="resulthead">
            <div>
              <div class="sectiontitle">提问包预览</div>
              <div class="pillnote" id="clipboard-path">等待生成</div>
            </div>
            <div class="btnrow">
              <div class="tokenbox">预估 <strong id="token-estimate">0</strong> tokens</div>
              <button class="secondary" type="button" id="copy-btn">一键复制</button>
            </div>
          </div>
          <div class="render">
            <article id="markdown-output">
              <div class="empty">这里会显示生成好的 markdown。</div>
            </article>
          </div>
        </section>
      </section>

      <section class="lower">
        <div class="stack">
          <section class="panel retrieved">
            <div class="sectiontitle">本次召回</div>
            <div class="quotelist" id="retrieved-list">
              <div class="empty">生成后会显示每条召回语录的 score、type、date 和摘要。</div>
            </div>
          </section>
          <section class="panel statuspanel">
            <div class="sectiontitle">状态与分布</div>
            <div class="dashboard">
              <div class="donut">
                <div>Domain</div>
                <div class="ring" id="domain-ring"></div>
                <div class="legend" id="domain-legend"></div>
              </div>
              <div class="donut">
                <div>Type</div>
                <div class="ring" id="type-ring"></div>
                <div class="legend" id="type-legend"></div>
              </div>
              <div class="donut">
                <div>Time Sensitivity</div>
                <div class="ring" id="time-ring"></div>
                <div class="legend" id="time-legend"></div>
              </div>
            </div>
            <pre class="statuspre" id="status-output"></pre>
          </section>
        </div>
        <section class="panel history">
          <div class="sectiontitle">最近 20 次</div>
          <div class="historylist" id="history-list">
            <div class="empty">还没有历史记录。</div>
          </div>
        </section>
      </section>
    </div>
  </div>
  <script>
    const palette = ["#d8a96b", "#6dc3b1", "#7e92ab", "#c98f7b", "#9ec36d", "#b899d0", "#d8d06b", "#6ba4d8"];
    const form = document.getElementById("generator-form");
    const questionInput = document.getElementById("question");
    const contextInput = document.getElementById("context");
    const personaInput = document.getElementById("persona-file");
    const topK = document.getElementById("top-k");
    const topKValue = document.getElementById("top-k-value");
    const markdownOutput = document.getElementById("markdown-output");
    const tokenEstimate = document.getElementById("token-estimate");
    const clipboardPath = document.getElementById("clipboard-path");
    const retrievedList = document.getElementById("retrieved-list");
    const historyList = document.getElementById("history-list");
    const historyCount = document.getElementById("history-count");
    const statusOutput = document.getElementById("status-output");
    const copyBtn = document.getElementById("copy-btn");
    const clearBtn = document.getElementById("clear-btn");
    const hint = document.getElementById("hint");

    topK.addEventListener("input", () => { topKValue.textContent = topK.value; });

    function checkedValues(containerId) {
      return [...document.querySelectorAll(`#${containerId} input:checked`)].map((el) => el.value);
    }

    function renderChecks(containerId, values) {
      const el = document.getElementById(containerId);
      if (!values.length) {
        el.innerHTML = '<div class="empty">没有可选项。</div>';
        return;
      }
      el.innerHTML = values.map((value) => `
        <label class="checkchip">
          <input type="checkbox" value="${value}">
          <span>${value}</span>
        </label>
      `).join("");
    }

    function renderLegend(containerId, counts) {
      const entries = Object.entries(counts);
      const el = document.getElementById(containerId);
      if (!entries.length) {
        el.innerHTML = '<div class="empty">暂无数据。</div>';
        return;
      }
      const total = entries.reduce((sum, [, count]) => sum + count, 0) || 1;
      el.innerHTML = entries.slice(0, 8).map(([label, count], index) => `
        <div class="legenditem">
          <span class="swatch" style="background:${palette[index % palette.length]}"></span>
          <span>${label}</span>
          <span>${count}</span>
        </div>
      `).join("");
      const stops = [];
      let acc = 0;
      entries.slice(0, 8).forEach(([_, count], index) => {
        const start = acc;
        acc += (count / total) * 100;
        stops.push(`${palette[index % palette.length]} ${start}% ${acc}%`);
      });
      const ringId = containerId.replace("legend", "ring");
      document.getElementById(ringId).style.background = `conic-gradient(${stops.join(", ")})`;
    }

    function renderHistory(history) {
      historyCount.textContent = history.length;
      if (!history.length) {
        historyList.innerHTML = '<div class="empty">还没有历史记录。</div>';
        return;
      }
      historyList.innerHTML = history.map((item, index) => `
        <button class="historybtn" type="button" data-index="${index}">
          <div>${item.question}</div>
          <div class="historymeta">${item.timestamp} · ${item.persona || "meta_thinking.md"} · top_k ${item.top_k} · ${item.token_estimate} tokens</div>
        </button>
      `).join("");
      [...historyList.querySelectorAll(".historybtn")].forEach((button) => {
        button.addEventListener("click", () => {
          const item = history[Number(button.dataset.index)];
          questionInput.value = item.question || "";
          contextInput.value = item.context || "";
          personaInput.value = item.persona || "meta_thinking.md";
          topK.value = item.top_k || 8;
          topK.dispatchEvent(new Event("input"));
          setChecked("domain-filters", item.domains || []);
          setChecked("type-filters", item.quote_types || []);
          setChecked("time-filters", item.time_sensitivities || []);
        });
      });
    }

    function setChecked(containerId, values) {
      const allow = new Set(values);
      [...document.querySelectorAll(`#${containerId} input`)].forEach((el) => { el.checked = allow.has(el.value); });
    }

    function renderRetrieved(rows) {
      if (!rows.length) {
        retrievedList.innerHTML = '<div class="empty">这次没有召回到结果。</div>';
        return;
      }
      retrievedList.innerHTML = rows.map((row) => `
        <div class="quoteitem">
          <div class="quotehead">
            <span>ID ${row.source_id}</span>
            <span>score ${row.score}</span>
            <span>${row.type}</span>
            <span>${row.date}</span>
            <span>${row.time_sensitivity}</span>
          </div>
          <div class="quotecontent">${escapeHtml(row.content)}</div>
          ${row.summary ? `<div class="quotesummary">${escapeHtml(row.summary)}</div>` : ""}
        </div>
      `).join("");
    }

    function escapeHtml(text) {
      return String(text)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;");
    }

    async function loadMeta() {
      const res = await fetch("/api/meta");
      const data = await res.json();
      renderChecks("domain-filters", data.domains || []);
      renderChecks("type-filters", data.quote_types || []);
      renderChecks("time-filters", data.time_sensitivities || []);
      renderLegend("domain-legend", data.domain_counts || {});
      renderLegend("type-legend", data.type_counts || {});
      renderLegend("time-legend", data.time_counts || {});
      statusOutput.textContent = data.status || "";
      renderHistory(data.history || []);
    }

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const payload = {
        question: questionInput.value.trim(),
        context: contextInput.value.trim(),
        persona: personaInput.value.trim() || "meta_thinking.md",
        domains: checkedValues("domain-filters"),
        quote_types: checkedValues("type-filters"),
        time_sensitivities: checkedValues("time-filters"),
        top_k: Number(topK.value),
      };
      if (!payload.question) {
        hint.textContent = "问题不能为空。";
        hint.classList.add("danger");
        return;
      }
      hint.textContent = "正在生成...";
      hint.classList.remove("danger");
      const res = await fetch("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (!res.ok) {
        hint.textContent = data.detail || "生成失败。";
        hint.classList.add("danger");
        return;
      }
      markdownOutput.innerHTML = marked.parse(data.markdown);
      tokenEstimate.textContent = data.token_estimate;
      clipboardPath.textContent = data.clipboard_path;
      renderRetrieved(data.retrieved || []);
      renderHistory(data.history || []);
      hint.textContent = "生成完成，结果已经同步写入 clipboard.md。";
      hint.classList.remove("danger");
      window.latestMarkdown = data.markdown;
    });

    copyBtn.addEventListener("click", async () => {
      const text = window.latestMarkdown || markdownOutput.innerText || "";
      await navigator.clipboard.writeText(text);
      hint.textContent = "已复制到剪贴板。";
      hint.classList.remove("danger");
    });

    clearBtn.addEventListener("click", () => {
      questionInput.value = "";
      contextInput.value = "";
      personaInput.value = "meta_thinking.md";
      topK.value = 8;
      topK.dispatchEvent(new Event("input"));
      setChecked("domain-filters", []);
      setChecked("type-filters", []);
      setChecked("time-filters", []);
    });

    loadMeta();
  </script>
</body>
</html>"""
