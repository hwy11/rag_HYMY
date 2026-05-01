const fields = {
  authorization: document.getElementById("authorization"),
  userAgent: document.getElementById("userAgent"),
  groupId: document.getElementById("groupId"),
  topicsUrl: document.getElementById("topicsUrl"),
  scope: document.getElementById("scope"),
  crawlMode: document.getElementById("crawlMode"),
  windowStartTime: document.getElementById("windowStartTime"),
  windowEndTime: document.getElementById("windowEndTime"),
  maxNewTopicsPerRun: document.getElementById("maxNewTopicsPerRun"),
  autoExport: document.getElementById("autoExport"),
};

const statusFields = {
  phase: document.getElementById("phase"),
  phaseDetail: document.getElementById("phaseDetail"),
  baselineTime: document.getElementById("baselineTime"),
  progressTime: document.getElementById("progressTime"),
  currentPageRange: document.getElementById("currentPageRange"),
  runMode: document.getElementById("runMode"),
  runLimit: document.getElementById("runLimit"),
  runGot: document.getElementById("runGot"),
};

const modeText = {
  after_baseline: "只补你现有数据之后的新内容",
  time_window: "只抓指定时间段里的内容",
  full_history: "继续往更早的旧内容回填",
};

const statusText = {
  idle: "等待开始",
  fetching: "正在抓取",
  sleeping: "按限流休息中",
  done: "抓取完成",
  error: "抓取失败",
};

let pollingTimer = null;

async function request(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });

  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.detail || payload.message || "请求失败");
  }
  return payload;
}

function renderStatus(status) {
  statusFields.phase.textContent = statusText[status.phase] || status.phase || "-";
  statusFields.phaseDetail.textContent = status.phase_detail || "-";
  statusFields.baselineTime.textContent = status.baseline_time || "-";
  statusFields.progressTime.textContent = status.progress_time || "-";
  statusFields.currentPageRange.textContent = status.current_page_range || "-";
  statusFields.runMode.textContent = modeText[status.run_mode] || status.run_mode || "-";
  statusFields.runLimit.textContent = status.run_limit || "-";
  statusFields.runGot.textContent = status.run_got ?? 0;
}

async function loadConfig() {
  const config = await request("/api/config");
  fields.authorization.value = config.authorization || "";
  fields.userAgent.value = config.user_agent || "";
  fields.groupId.value = config.group_id || "";
  fields.topicsUrl.value = config.topics_url || "";
  fields.scope.value = config.scope || "";
  fields.crawlMode.value = config.crawl_mode || "after_baseline";
  fields.windowStartTime.value = config.window_start_time || "";
  fields.windowEndTime.value = config.window_end_time || "";
  fields.maxNewTopicsPerRun.value = config.max_new_topics_per_run ?? 50;
  fields.autoExport.value = String(config.auto_export ?? true);
}

async function saveConfig() {
  const payload = {
    authorization: fields.authorization.value.trim(),
    user_agent: fields.userAgent.value.trim(),
    group_id: fields.groupId.value.trim(),
    topics_url: fields.topicsUrl.value.trim(),
    scope: fields.scope.value.trim(),
    crawl_mode: fields.crawlMode.value,
    window_start_time: fields.windowStartTime.value.trim(),
    window_end_time: fields.windowEndTime.value.trim(),
    max_new_topics_per_run: Number(fields.maxNewTopicsPerRun.value || 50),
    auto_export: fields.autoExport.value === "true",
  };
  await request("/api/config", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

async function testConnection() {
  const result = await request("/api/test-connection", { method: "POST" });
  alert(`连上了。当前接口能拿到 ${result.sample_count} 条样本，最新一条时间是 ${result.latest_topic_time || "未知"}。`);
}

async function startCrawl() {
  await saveConfig();
  statusFields.phase.textContent = "正在抓取";
  statusFields.phaseDetail.textContent = "正在准备抓取";
  await request("/api/crawl", { method: "POST" });
  await refreshStatus();
}

async function refreshStatus() {
  const status = await request("/api/status");
  renderStatus(status);
}

document.getElementById("saveConfig").addEventListener("click", () => {
  saveConfig()
    .then(() => {
      statusFields.phaseDetail.textContent = "配置已保存到本机";
    })
    .catch((error) => alert(error.message));
});

document.getElementById("testConnection").addEventListener("click", () => {
  saveConfig()
    .then(() => testConnection())
    .catch((error) => alert(error.message));
});

document.getElementById("startCrawl").addEventListener("click", () => {
  startCrawl().catch((error) => alert(error.message));
});

loadConfig().catch(console.error);
refreshStatus().catch(console.error);
pollingTimer = setInterval(() => refreshStatus().catch(console.error), 800);
