const state = {
  runId: null,
  scriptDirty: false,
  lastAiScript: "",
  tasks: {},
  runPoller: null,
  taskPoller: null,
};

const el = {
  topic: document.getElementById("topic"),
  outline: document.getElementById("outline"),
  outlineFeedback: document.getElementById("outline-feedback"),
  script: document.getElementById("script"),
  scriptFeedback: document.getElementById("script-feedback"),
  outlineGenerate: document.getElementById("outline-generate"),
  outlineRevise: document.getElementById("outline-revise"),
  startRun: document.getElementById("start-run"),
  saveScript: document.getElementById("save-script"),
  validateScript: document.getElementById("validate-script"),
  reviseScript: document.getElementById("revise-script"),
  polishScript: document.getElementById("polish-script"),
  continueRun: document.getElementById("continue-run"),
  applyAi: document.getElementById("apply-ai"),
  aiOutput: document.getElementById("ai-output"),
  logs: document.getElementById("logs"),
  tasks: document.getElementById("tasks"),
  outputs: document.getElementById("outputs"),
  globalStatus: document.getElementById("global-status"),
  runId: document.getElementById("run-id"),
  runStatus: document.getElementById("run-status"),
  runStage: document.getElementById("run-stage"),
};

function setGlobalStatus(text) {
  el.globalStatus.textContent = text;
}

async function apiPost(path, payload) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload || {}),
  });
  const data = await res.json();
  if (!res.ok) {
    throw new Error(data.error || "Request failed");
  }
  return data;
}

async function apiGet(path) {
  const res = await fetch(path);
  const data = await res.json();
  if (!res.ok) {
    throw new Error(data.error || "Request failed");
  }
  return data;
}

function addLocalLog(message, level) {
  const item = document.createElement("div");
  item.className = "log-item";
  item.innerHTML = `<strong>${level || "info"}</strong> ${message}`;
  el.logs.prepend(item);
}

function renderLogs(events) {
  if (!events || events.length === 0) {
    el.logs.innerHTML = "<div class=\"muted\">No activity yet.</div>";
    return;
  }
  el.logs.innerHTML = "";
  events.slice().reverse().forEach((evt) => {
    const div = document.createElement("div");
    div.className = "log-item";
    div.innerHTML = `<strong>${evt.level}</strong> ${evt.ts} - ${evt.message}`;
    el.logs.appendChild(div);
  });
}

function renderTasks() {
  const taskList = Object.values(state.tasks);
  if (taskList.length === 0) {
    el.tasks.innerHTML = "<div class=\"muted\">No AI tasks yet.</div>";
    return;
  }
  el.tasks.innerHTML = "";
  taskList.slice().reverse().forEach((task) => {
    const div = document.createElement("div");
    div.className = "task-item";
    div.innerHTML = `<span>${task.type}</span><span class=\"task-status\">${task.status}</span>`;
    el.tasks.appendChild(div);
  });
}

function renderOutputs(outputs) {
  if (!outputs) {
    el.outputs.innerHTML = "<div class=\"muted\">Outputs will appear after the run completes.</div>";
    return;
  }
  el.outputs.innerHTML = "";
  const dir = document.createElement("div");
  dir.className = "file";
  dir.innerHTML = `<span>Output dir</span><span>${outputs.run_dir}</span>`;
  el.outputs.appendChild(dir);
  if (outputs.image_count !== undefined) {
    const img = document.createElement("div");
    img.className = "file";
    img.innerHTML = `<span>Images</span><span>${outputs.image_count}</span>`;
    el.outputs.appendChild(img);
  }
  (outputs.files || []).forEach((file) => {
    const row = document.createElement("div");
    row.className = "file";
    row.innerHTML = `<span>${file}</span><span>ready</span>`;
    el.outputs.appendChild(row);
  });
}

function setAiOutput(text, isScript) {
  el.aiOutput.textContent = text || "";
  state.lastAiScript = isScript ? text : "";
  el.applyAi.disabled = !state.lastAiScript;
}

function addTask(taskId, type) {
  state.tasks[taskId] = { taskId, type, status: "running" };
  renderTasks();
}

async function pollTasks() {
  const running = Object.values(state.tasks).filter((t) => t.status === "running");
  if (running.length === 0) {
    return;
  }
  for (const task of running) {
    try {
      const data = await apiGet(`/api/task/${task.taskId}`);
      state.tasks[task.taskId] = data;
      if (data.status === "done") {
        handleTaskResult(data);
      }
      if (data.status === "error") {
        setAiOutput(`Task failed: ${data.error}`, false);
      }
    } catch (err) {
      addLocalLog(err.message, "error");
    }
  }
  renderTasks();
}

function handleTaskResult(task) {
  if (task.type === "outline.generate" || task.type === "outline.revise") {
    el.outline.value = task.result || "";
    return;
  }
  if (task.type === "script.validate") {
    setAiOutput(JSON.stringify(task.result, null, 2), false);
    return;
  }
  if (task.type === "script.revise" || task.type === "script.polish") {
    setAiOutput(task.result || "", true);
  }
}

async function pollRunStatus() {
  if (!state.runId) {
    return;
  }
  try {
    const data = await apiGet(`/api/run/status/${state.runId}`);
    el.runId.textContent = data.run_id || "-";
    el.runStatus.textContent = data.status || "-";
    el.runStage.textContent = data.stage || "-";
    setGlobalStatus(data.status || "Idle");
    renderLogs(data.events || []);

    if (!state.scriptDirty && data.script) {
      el.script.value = data.script;
    }

    if (data.status === "awaiting_review") {
      el.continueRun.disabled = false;
    }
    if (data.status === "running_post_script" || data.status === "running_pre_script") {
      el.continueRun.disabled = true;
    }
    if (data.status === "complete") {
      renderOutputs(data.outputs);
    }
    if (data.status === "error") {
      setAiOutput(`Run error: ${data.error || "unknown"}`, false);
    }
  } catch (err) {
    addLocalLog(err.message, "error");
  }
}

function startPolling() {
  if (!state.runPoller) {
    state.runPoller = setInterval(pollRunStatus, 1500);
  }
  if (!state.taskPoller) {
    state.taskPoller = setInterval(pollTasks, 1200);
  }
}

el.script.addEventListener("input", () => {
  state.scriptDirty = true;
});

el.applyAi.addEventListener("click", () => {
  if (!state.lastAiScript) {
    return;
  }
  el.script.value = state.lastAiScript;
  state.scriptDirty = true;
});

el.outlineGenerate.addEventListener("click", async () => {
  try {
    const topic = el.topic.value.trim();
    if (!topic) {
      addLocalLog("Topic is required for outline generation", "warn");
      return;
    }
    const data = await apiPost("/api/outline/generate", { topic });
    addTask(data.task_id, "outline.generate");
  } catch (err) {
    addLocalLog(err.message, "error");
  }
});

el.outlineRevise.addEventListener("click", async () => {
  try {
    const topic = el.topic.value.trim();
    const outline = el.outline.value.trim();
    const feedback = el.outlineFeedback.value.trim();
    if (!topic || !outline || !feedback) {
      addLocalLog("Topic, outline, and feedback are required", "warn");
      return;
    }
    const data = await apiPost("/api/outline/revise", { topic, outline, feedback });
    addTask(data.task_id, "outline.revise");
  } catch (err) {
    addLocalLog(err.message, "error");
  }
});

el.startRun.addEventListener("click", async () => {
  try {
    const topic = el.topic.value.trim();
    const outline = el.outline.value.trim();
    if (!topic) {
      addLocalLog("Topic is required to start a run", "warn");
      return;
    }
    const data = await apiPost("/api/run/start", { topic, outline });
    state.runId = data.run_id;
    state.scriptDirty = false;
    setGlobalStatus("running");
    startPolling();
  } catch (err) {
    addLocalLog(err.message, "error");
  }
});

el.saveScript.addEventListener("click", async () => {
  try {
    if (!state.runId) {
      addLocalLog("Start a run before saving script", "warn");
      return;
    }
    const script = el.script.value.trim();
    await apiPost("/api/run/update_script", { run_id: state.runId, script });
    state.scriptDirty = false;
    addLocalLog("Script saved to run", "info");
  } catch (err) {
    addLocalLog(err.message, "error");
  }
});

el.validateScript.addEventListener("click", async () => {
  try {
    const script = el.script.value.trim();
    if (!script) {
      addLocalLog("Script is required for validation", "warn");
      return;
    }
    const data = await apiPost("/api/script/validate", { script });
    addTask(data.task_id, "script.validate");
  } catch (err) {
    addLocalLog(err.message, "error");
  }
});

el.reviseScript.addEventListener("click", async () => {
  try {
    const script = el.script.value.trim();
    const feedback = el.scriptFeedback.value.trim();
    if (!script || !feedback) {
      addLocalLog("Script and feedback are required", "warn");
      return;
    }
    const payload = {
      script,
      feedback,
      run_id: state.runId,
      topic: el.topic.value.trim(),
      outline: el.outline.value.trim(),
    };
    const data = await apiPost("/api/script/revise", payload);
    addTask(data.task_id, "script.revise");
  } catch (err) {
    addLocalLog(err.message, "error");
  }
});

el.polishScript.addEventListener("click", async () => {
  try {
    const script = el.script.value.trim();
    if (!script) {
      addLocalLog("Script is required", "warn");
      return;
    }
    const payload = {
      script,
      run_id: state.runId,
      topic: el.topic.value.trim(),
      outline: el.outline.value.trim(),
    };
    const data = await apiPost("/api/script/polish", payload);
    addTask(data.task_id, "script.polish");
  } catch (err) {
    addLocalLog(err.message, "error");
  }
});

el.continueRun.addEventListener("click", async () => {
  try {
    if (!state.runId) {
      addLocalLog("Start a run first", "warn");
      return;
    }
    if (state.scriptDirty) {
      await apiPost("/api/run/update_script", {
        run_id: state.runId,
        script: el.script.value.trim(),
      });
      state.scriptDirty = false;
    }
    await apiPost("/api/run/continue", { run_id: state.runId });
    el.continueRun.disabled = true;
    addLocalLog("Continuing workflow", "info");
  } catch (err) {
    addLocalLog(err.message, "error");
  }
});

startPolling();
