document.addEventListener("submit", async (event) => {
  const form = event.target;
  if (!(form instanceof HTMLFormElement)) {
    return;
  }

  const confirmMessage = form.getAttribute("data-confirm") || form.dataset.confirm;
  if (confirmMessage) {
    if (!window.confirm(confirmMessage)) {
      event.preventDefault();
      return;
    }
  }

  if (form.matches("[data-export-form]")) {
    await submitExportForm(event, form);
    return;
  }

  if (!form.matches("[data-regenerate-form]")) {
    return;
  }

  event.preventDefault();

  const block = form.closest("[data-result-block]");
  const button = form.querySelector("button[type='submit']");
  if (!(block instanceof HTMLElement)) {
    form.submit();
    return;
  }

  if (button instanceof HTMLButtonElement) {
    button.disabled = true;
    button.textContent = "Обновляется";
  }
  block.setAttribute("aria-busy", "true");
  showTaskState(block, "Повторная генерация поставлена в очередь.");

  try {
    const response = await fetch(form.action, {
      method: "POST",
      headers: {
        "Accept": "text/html",
        "X-Linguary-Fragment": "result-block",
      },
      credentials: "same-origin",
    });
    if (!response.ok) {
      throw new Error(`Request failed: ${response.status}`);
    }
    const template = document.createElement("template");
    template.innerHTML = (await response.text()).trim();
    const replacement = template.content.firstElementChild;
    if (!(replacement instanceof HTMLElement)) {
      throw new Error("Empty result block response");
    }
    block.replaceWith(replacement);
    replacement.scrollIntoView({ block: "nearest" });
    pollResultBlock(replacement);
    refreshResultHeader();
  } catch (_error) {
    block.removeAttribute("aria-busy");
    showTaskState(block, "Не удалось запустить обновление. Попробуйте ещё раз.");
    if (button instanceof HTMLButtonElement) {
      button.disabled = false;
      button.textContent = "Обновить";
    }
  }
});

async function submitExportForm(event, form) {
  event.preventDefault();

  const header = form.closest("[data-result-header]");
  const button = form.querySelector("button[type='submit']");
  if (button instanceof HTMLButtonElement) {
    button.disabled = true;
    button.textContent = "Собирается";
  }
  if (header instanceof HTMLElement) {
    showHeaderState(header, "Сборка EPUB поставлена в очередь.");
  }

  try {
    const response = await fetch(form.action, {
      method: "POST",
      headers: { "Accept": "application/json" },
      credentials: "same-origin",
    });
    if (!response.ok) {
      throw new Error(`Request failed: ${response.status}`);
    }
    await response.json();
    await refreshResultHeader();
    pollResultHeader();
  } catch (_error) {
    if (header instanceof HTMLElement) {
      showHeaderState(header, "Не удалось запустить сборку EPUB. Попробуйте ещё раз.");
    }
    if (button instanceof HTMLButtonElement) {
      button.disabled = false;
      button.textContent = "Собрать EPUB";
    }
  }
}

function showTaskState(block, text) {
  let state = block.querySelector(".task-state");
  if (!(state instanceof HTMLElement)) {
    state = document.createElement("p");
    state.className = "task-state";
    state.setAttribute("role", "status");
    block.append(state);
  }
  state.textContent = text;
}

function showHeaderState(header, text) {
  const paragraph = header.querySelector("h1 + p");
  if (paragraph instanceof HTMLElement) {
    paragraph.setAttribute("role", "status");
    paragraph.textContent = text;
  }
}

function pollResultBlock(block) {
  const form = block.querySelector("[data-regenerate-form]");
  const state = block.querySelector("[data-active-task='true']");
  if (!(form instanceof HTMLFormElement) || !(state instanceof HTMLElement)) {
    return;
  }

  const url = form.action.replace(/\/regenerate$/, "");
  window.setTimeout(() => refreshResultBlock(block, url, 1), 1500);
}

async function refreshResultBlock(block, url, attempt) {
  if (!block.isConnected || attempt > 120) {
    return;
  }

  try {
    const response = await fetch(url, {
      method: "GET",
      headers: {
        "Accept": "text/html",
        "X-Linguary-Fragment": "result-block",
      },
      credentials: "same-origin",
    });
    if (!response.ok) {
      throw new Error(`Request failed: ${response.status}`);
    }
    const template = document.createElement("template");
    template.innerHTML = (await response.text()).trim();
    const replacement = template.content.firstElementChild;
    if (!(replacement instanceof HTMLElement)) {
      throw new Error("Empty result block response");
    }
    block.replaceWith(replacement);
    pollResultBlock(replacement);
    if (!replacement.querySelector("[data-active-task='true']")) {
      refreshResultHeader();
    }
  } catch (_error) {
    window.setTimeout(() => refreshResultBlock(block, url, attempt + 1), 2000);
  }
}

async function refreshResultHeader() {
  const header = document.querySelector("[data-result-header]");
  if (!(header instanceof HTMLElement)) {
    return null;
  }

  const url = header.dataset.resultUrl;
  if (!url) {
    return null;
  }

  const response = await fetch(url, {
    method: "GET",
    headers: {
      "Accept": "text/html",
      "X-Linguary-Fragment": "result-header",
    },
    credentials: "same-origin",
  });
  if (!response.ok) {
    throw new Error(`Request failed: ${response.status}`);
  }

  const template = document.createElement("template");
  template.innerHTML = (await response.text()).trim();
  const replacement = template.content.firstElementChild;
  if (!(replacement instanceof HTMLElement)) {
    throw new Error("Empty result header response");
  }

  header.replaceWith(replacement);
  return replacement;
}

function pollResultHeader(attempt = 1) {
  if (attempt > 120) {
    return;
  }

  window.setTimeout(async () => {
    try {
      const header = await refreshResultHeader();
      if (
        header instanceof HTMLElement &&
        header.querySelector("[data-export-form] button:disabled")
      ) {
        pollResultHeader(attempt + 1);
      }
    } catch (_error) {
      pollResultHeader(attempt + 1);
    }
  }, 1500);
}

function initSettingsForm() {
  const form = document.querySelector(".settings-form");
  if (!(form instanceof HTMLFormElement)) {
    return;
  }

  const proxyModeSelect = document.getElementById("settings-proxy-mode");
  const proxyUrlContainer = document.getElementById("settings-field-proxy-url");
  const proxyUrlInput = document.getElementById("settings-proxy-url");

  function updateProxyVisibility() {
    if (!proxyModeSelect || !proxyUrlContainer) {
      return;
    }
    const isCustom = proxyModeSelect.value === "custom";
    proxyUrlContainer.hidden = !isCustom;
    proxyUrlContainer.style.display = isCustom ? "" : "none";
    if (proxyUrlInput instanceof HTMLInputElement) {
      proxyUrlInput.disabled = !isCustom;
    }
  }

  if (proxyModeSelect instanceof HTMLSelectElement) {
    proxyModeSelect.addEventListener("change", updateProxyVisibility);
    updateProxyVisibility();
  }

  const testButton = document.getElementById("settings-test-connection");
  if (testButton instanceof HTMLButtonElement) {
    testButton.addEventListener("click", testLLMConnection);
  }
}

async function testLLMConnection() {
  const button = document.getElementById("settings-test-connection");
  const status = document.getElementById("settings-test-status");
  if (!(button instanceof HTMLButtonElement) || !(status instanceof HTMLElement)) {
    return;
  }

  const form = button.closest("form");
  if (!(form instanceof HTMLFormElement)) {
    return;
  }

  const getVal = (name) => {
    const el = form.elements.namedItem(name);
    if (!el) return undefined;
    if (el instanceof HTMLInputElement || el instanceof HTMLSelectElement) {
      return el.value;
    }
    return undefined;
  };

  const payload = {
    provider: getVal("provider"),
    base_url: getVal("base_url"),
    model: getVal("model"),
    analysis_model: getVal("analysis_model"),
    processing_model: getVal("processing_model"),
    api_key: getVal("api_key"),
    proxy_mode: getVal("proxy_mode"),
    proxy_url: getVal("proxy_url"),
  };

  button.disabled = true;
  status.className = "settings-test-status";
  status.textContent = "Проверка подключения...";

  try {
    const response = await fetch("/system/config/test", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Accept": "application/json",
      },
      body: JSON.stringify(payload),
      credentials: "same-origin",
    });

    const data = await response.json();
    if (response.ok && data.ok) {
      status.className = "settings-test-status ok";
      status.textContent = data.message || "Подключение успешно установлено.";
    } else {
      status.className = "settings-test-status error";
      status.textContent = data.error || data.message || "Ошибка подключения.";
    }
  } catch (_error) {
    status.className = "settings-test-status error";
    status.textContent = "Не удалось выполнить проверку связи.";
  } finally {
    button.disabled = false;
  }
}

window.initSettingsForm = initSettingsForm;
window.testLLMConnection = testLLMConnection;

document.addEventListener("DOMContentLoaded", () => {
  initSettingsForm();
  document.querySelectorAll("[data-result-block]").forEach((block) => {
    if (block instanceof HTMLElement) {
      pollResultBlock(block);
    }
  });
});
