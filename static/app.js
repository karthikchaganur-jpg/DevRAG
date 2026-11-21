const $ = (selector) => document.querySelector(selector);

let indexed = false;
let asking = false;
let toastTimer;
let currentRepoId = null;
let currentView = "hero";

function toast(message) {
  const element = $("#toast");
  element.textContent = message;
  element.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => element.classList.remove("show"), 3000);
}

async function request(url, options = {}) {
  const response = await fetch(url, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
  });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.detail || "Something went wrong.");
  return payload;
}

const post = (url, data) => request(url, { method: "POST", body: JSON.stringify(data) });

function setBusy(button, busy, label) {
  button.disabled = busy;
  if (busy) {
    button.dataset.label = button.textContent;
    button.textContent = label;
  } else if (button.dataset.label) {
    button.textContent = button.dataset.label;
    delete button.dataset.label;
  }
}

function setIndexed(info, repoId = currentRepoId) {
  indexed = true;
  currentRepoId = repoId;
  const repoName = info.root.split(/[\\/]/).filter(Boolean).pop() || info.root;
  $("#current-repo-name").textContent = repoName;
  $("#file-count").textContent = info.files;
  $("#status-title").textContent = "Repository indexed";
  $("#status-subtitle").textContent = `${info.files} files · ${info.chunks} code sections`;
  $("#status-pulse").classList.add("ready");
  $("#files-root").textContent = info.root;
}

function renderRepositories(repos, selectedId) {
  const list = $("#repo-list");
  list.replaceChildren();
  const selected = repos.find((repo) => repo.id === selectedId);
  $("#current-repo-name").textContent = selected?.name || "No repository selected";
  if (!repos.length) {
    const empty = document.createElement("div");
    empty.className = "repo-list-empty";
    empty.textContent = "No saved repositories";
    list.append(empty);
    return;
  }
  for (const repo of repos) {
    const button = document.createElement("button");
    button.className = `repo-choice${repo.id === selectedId ? " active" : ""}`;
    button.type = "button";
    button.dataset.repoId = repo.id;
    button.title = repo.path;
    const dot = document.createElement("span");
    dot.className = "repo-choice-dot";
    dot.setAttribute("aria-hidden", "true");
    const name = document.createElement("span");
    name.className = "repo-choice-name";
    name.textContent = repo.name;
    button.append(dot, name);
    button.addEventListener("click", () => selectRepository(repo.id));
    list.append(button);
  }
}

function clearConversation() {
  $("#chat-feed").replaceChildren();
}

function showView(view) {
  currentView = view;
  $("#hero").classList.toggle("hidden", view !== "hero");
  $("#chat-view").classList.toggle("hidden", view !== "chat");
  $("#files-view").classList.toggle("hidden", view !== "files");
  if (view === "files") loadFiles();
}

async function addRepository(button) {
  const path = $("#repo-path").value.trim();
  if (!path) {
    toast("Enter a repository folder path first.");
    $("#repo-path").focus();
    return;
  }
  setBusy(button, true, "Indexing…");
  try {
    const result = await post("/api/repos", { path });
    renderRepositories(result.repos, result.current_id);
    currentRepoId = result.current_id;
    setIndexed(result.info, result.current_id);
    clearConversation();
    $("#repo-path").value = "";
    showView("hero");
    toast(`Added ${result.repo.name}`);
  } catch (error) {
    toast(error.message);
  } finally {
    setBusy(button, false);
  }
}

async function selectRepository(repoId) {
  if (!repoId || (repoId === currentRepoId && indexed)) return;
  document.querySelectorAll(".repo-choice").forEach((button) => { button.disabled = true; });
  try {
    const result = await post("/api/repos/select", { id: repoId });
    currentRepoId = repoId;
    setIndexed(result.info, repoId);
    document.querySelectorAll(".repo-choice").forEach((button) => {
      button.classList.toggle("active", button.dataset.repoId === repoId);
    });
    clearConversation();
    showView("hero");
    toast(`Switched to ${result.repo.name}`);
  } catch (error) {
    toast(error.message);
    await loadRepositories();
  } finally {
    document.querySelectorAll(".repo-choice").forEach((button) => { button.disabled = false; });
  }
}

async function loadRepositories() {
  try {
    const result = await request("/api/repos");
    renderRepositories(result.repos, result.current_id);
    if (result.current_id) {
      const selected = await post("/api/repos/select", { id: result.current_id });
      setIndexed(selected.info, result.current_id);
    } else {
      indexed = false;
      $("#status-subtitle").textContent = "Select or add a repository";
    }
  } catch (error) {
    toast(error.message);
  }
}

async function loadFiles() {
  const list = $("#file-list");
  list.replaceChildren();
  $("#file-content").textContent = "Choose a file to preview its contents.";
  $("#file-title").textContent = "Choose a file";
  $("#file-lines").textContent = "";
  if (!indexed) {
    list.innerHTML = '<div class="file-empty">Select a repository to browse its files.</div>';
    return;
  }
  try {
    const result = await request("/api/files");
    $("#files-root").textContent = result.root;
    if (!result.files.length) {
      list.innerHTML = '<div class="file-empty">No supported source files found.</div>';
      return;
    }
    for (const file of result.files) {
      const button = document.createElement("button");
      button.className = "file-entry";
      button.type = "button";
      button.dataset.path = file.path;
      const name = document.createElement("span");
      name.textContent = file.path;
      const lines = document.createElement("small");
      lines.textContent = `${file.lines} lines`;
      button.append(name, lines);
      button.addEventListener("click", () => openFile(file, button));
      list.append(button);
    }
  } catch (error) {
    list.innerHTML = '<div class="file-empty">Could not load the source file list.</div>';
    toast(error.message);
  }
}

async function openFile(file, button) {
  document.querySelectorAll(".file-entry.selected").forEach((item) => item.classList.remove("selected"));
  button.classList.add("selected");
  $("#file-title").textContent = file.path;
  $("#file-lines").textContent = `${file.lines} lines`;
  $("#file-content").textContent = "Loading file…";
  try {
    const result = await request(`/api/files/content?path=${encodeURIComponent(file.path)}`);
    $("#file-content").textContent = result.content;
  } catch (error) {
    $("#file-content").textContent = error.message;
  }
}

function escapeHtml(value) {
  return value.replace(/[&<>"']/g, (character) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[character]);
}

function answerMarkup(text) {
  const safe = escapeHtml(text)
    .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
    .replace(/`([^`]+)`/g, "<code>$1</code>");
  return safe.split("\n").map((line) => {
    const match = line.match(/^\d+\.\s+(.*)$/);
    if (match) return `<li>${match[1]}</li>`;
    if (!line.trim()) return "";
    return `<p>${line}</p>`;
  }).join("").replace(/(<li>.*?<\/li>)+/g, "<ol>$&</ol>");
}

function renderAnswer(result) {
  const card = document.createElement("article");
  card.className = "answer-card";
  const sources = result.sources.map((source) => `
    <div class="source-item">
      <span class="source-path">${escapeHtml(source.path)}:${source.match_lines?.length ? source.match_lines.join(", ") : `${source.start}–${source.end}`}</span>
      <pre>${escapeHtml(source.text)}</pre>
    </div>`).join("");
  card.innerHTML = `
    <div class="answer-mark">✳</div>
    <div class="answer-body">
      ${answerMarkup(result.answer)}
      <div class="answer-meta">${result.sources.length} source${result.sources.length === 1 ? "" : "s"} retrieved</div>
      <div class="sources"><div class="sources-head">REFERENCES</div>${sources || '<div class="source-item">No matching source excerpts</div>'}</div>
    </div>`;
  $("#chat-feed").append(card);
  $("#content").scrollTo({ top: $("#content").scrollHeight, behavior: "smooth" });
}

async function ask(question) {
  if (asking) return;
  if (!indexed) {
    toast("Select or add a repository first");
    return;
  }
  asking = true;
  showView("chat");
  const user = document.createElement("div");
  user.className = "user-question";
  user.textContent = question;
  const wait = document.createElement("div");
  wait.className = "answer-card";
  wait.innerHTML = '<div class="answer-mark">✳</div><div class="answer-body loading">Searching your codebase<span class="dots"></span></div>';
  $("#chat-feed").append(user, wait);
  $("#content").scrollTo({ top: $("#content").scrollHeight, behavior: "smooth" });
  setBusy($("#send-button"), true, "…");
  $("#question").disabled = true;
  try {
    const result = await post("/api/ask", { question, limit: 6 });
    wait.remove();
    renderAnswer(result);
  } catch (error) {
    wait.remove();
    toast(error.message);
  } finally {
    asking = false;
    $("#question").disabled = false;
    setBusy($("#send-button"), false);
    $("#question").focus();
  }
}

$("#index-button").addEventListener("click", () => {
  showView("hero");
  $("#repo-path").focus();
});
$("#hero-index").addEventListener("click", () => addRepository($("#hero-index")));
$("#repo-collapse").addEventListener("click", (event) => {
  const list = $("#repo-list");
  const collapsed = list.classList.toggle("collapsed");
  event.currentTarget.setAttribute("aria-expanded", String(!collapsed));
  event.currentTarget.setAttribute("aria-label", collapsed ? "Expand repository list" : "Collapse repository list");
  event.currentTarget.querySelector("span").textContent = collapsed ? "›" : "⌄";
});
$("#files-nav").addEventListener("click", () => showView("files"));
$("#back-to-ask").addEventListener("click", () => showView($("#chat-feed").children.length ? "chat" : "hero"));
$("#ask-form").addEventListener("submit", (event) => {
  event.preventDefault();
  const input = $("#question");
  const value = input.value.trim();
  if (value) {
    input.value = "";
    input.style.height = "auto";
    ask(value);
  }
});
$("#question").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    $("#ask-form").requestSubmit();
  }
});
$("#question").addEventListener("input", (event) => {
  event.target.style.height = "auto";
  event.target.style.height = `${Math.min(event.target.scrollHeight, 110)}px`;
});
document.querySelectorAll(".suggestion").forEach((button) => button.addEventListener("click", () => ask(button.dataset.question)));

loadRepositories();
