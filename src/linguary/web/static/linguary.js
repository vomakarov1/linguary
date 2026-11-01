let resultGroupsGeneration = 0;
let resultGroupsRequest = null;
let resultGroupsTimer = null;
let resultGroupsAttempt = 1;
let resultGroupsRefreshNeeded = false;
let resultGroupsRetryDelay = 1500;
let resultGroupsPendingFocus = null;

document.addEventListener("submit", async (event) => {
  const form = event.target;
  if (!(form instanceof HTMLFormElement)) {
    return;
  }

  if (!(form.getAttribute("onsubmit") && form.getAttribute("onsubmit").includes("confirm"))) {
    const confirmMessage = form.getAttribute("data-confirm") || form.dataset.confirm;
    if (confirmMessage) {
      if (!window.confirm(confirmMessage)) {
        event.preventDefault();
        return;
      }
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
  const blockId = form.dataset.blockId;
  const button = form.querySelector("button[type='submit']");
  if (!(block instanceof HTMLElement) || !blockId) {
    form.submit();
    return;
  }

  resultGroupsPendingFocus = captureResultFocus(block.closest("[data-result-groups]"));
  if (button instanceof HTMLButtonElement) {
    button.disabled = true;
    button.textContent = "Обновляется";
  }
  block.setAttribute("aria-busy", "true");
  showTaskState(block, "Повторная генерация поставлена в очередь.");

  resultGroupsGeneration += 1;
  window.clearTimeout(resultGroupsTimer);
  resultGroupsTimer = null;
  resultGroupsAttempt = 1;
  let submitted = false;
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
    submitted = true;
    resultGroupsGeneration += 1;
    resultGroupsRefreshNeeded = true;
    const root = document.querySelector("[data-result-groups]");
    if (root instanceof HTMLElement) {
      replaceResultGroup(root, blockId, await response.text());
    }
    refreshResultHeader().catch(() => {});
  } catch (_error) {
    if (block.isConnected) {
      block.removeAttribute("aria-busy");
      showTaskState(block, submitted
        ? "Не удалось обновить результат. Повторим попытку автоматически."
        : "Не удалось запустить обновление. Попробуйте ещё раз.");
      if (button instanceof HTMLButtonElement) {
        button.disabled = false;
        button.textContent = "Обновить";
      }
    }
  } finally {
    pollResultGroups(resultGroupsAttempt);
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

function captureResultFocus(root) {
  const active = document.activeElement;
  if (!(root instanceof HTMLElement) || !(active instanceof HTMLElement) || !root.contains(active)) {
    return null;
  }
  const block = active.closest("[data-result-block]");
  const form = active.closest("[data-regenerate-form]");
  return {
    blockId: block?.id,
    formAction: form?.action,
    href: active.closest("a")?.getAttribute("href"),
    id: active.id,
    guide: active.matches("details.ipa-guide > summary"),
  };
}

function restoreResultFocus(root, target) {
  if (!target) {
    return;
  }
  const block = Array.from(root.querySelectorAll("[data-result-block]")).find(
    (item) => item.id === target.blockId,
  );
  let element;
  if (target.formAction) {
    const form = Array.from(root.querySelectorAll("[data-regenerate-form]")).find(
      (item) => item.action === target.formAction,
    );
    element = form?.querySelector("button:not(:disabled)");
  } else if (target.href) {
    element = Array.from(root.querySelectorAll("a[href]")).find(
      (item) => item.getAttribute("href") === target.href,
    );
  } else if (target.id) {
    element = Array.from(root.querySelectorAll("[id]")).find((item) => item.id === target.id);
  } else if (target.guide) {
    element = block?.querySelector("details.ipa-guide > summary");
  }
  element = element || block;
  if (element instanceof HTMLElement) {
    if (element === block) {
      element.tabIndex = -1;
    }
    element.focus({ preventScroll: true });
  }
}

function resultGroupForBlock(root, blockId) {
  const form = Array.from(root.querySelectorAll("[data-regenerate-form]"))
    .find((item) => item.dataset.blockId === blockId);
  return form?.closest("[data-result-block]") || null;
}

function replaceResultGroup(root, blockId, markup) {
  const current = resultGroupForBlock(root, blockId);
  if (!(current instanceof HTMLElement)) {
    throw new Error("Result block was detached");
  }

  const template = document.createElement("template");
  template.innerHTML = markup.trim();
  const replacement = template.content.firstElementChild;
  if (
    !(replacement instanceof HTMLElement) ||
    template.content.children.length !== 1 ||
    template.content.querySelectorAll("[data-result-block]").length !== 1 ||
    !replacement.matches("[data-result-block]") || replacement.id !== current.id
  ) {
    throw new Error("Invalid result block response");
  }

  const scrollX = window.scrollX;
  const scrollY = window.scrollY;
  const focus = captureResultFocus(root);
  current.querySelectorAll("details.ipa-guide").forEach((guide, index) => {
    const nextGuide = replacement.querySelectorAll("details.ipa-guide")[index];
    if (nextGuide) {
      nextGuide.open = guide.open;
    }
  });
  current.replaceWith(replacement);
  restoreResultFocus(root, focus);
  window.scrollTo(scrollX, scrollY);
  return replacement;
}

async function refreshResultBlock(blockId) {
  const root = document.querySelector("[data-result-groups]");
  if (!(root instanceof HTMLElement)) {
    throw new Error("Missing result groups root");
  }
  const group = resultGroupForBlock(root, blockId);
  const form = Array.from(group?.querySelectorAll("[data-regenerate-form]") || [])
    .find((item) => item.dataset.blockId === blockId);
  if (!form) {
    throw new Error("Missing result block form");
  }

  const url = new URL(form.action, window.location.href);
  url.pathname = url.pathname.replace(/\/regenerate$/, "");
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

  const currentRoot = document.querySelector("[data-result-groups]");
  if (currentRoot !== root) {
    return null;
  }
  return replaceResultGroup(root, blockId, await response.text());
}

async function refreshActiveResultBlocks() {
  const root = document.querySelector("[data-result-groups]");
  if (!(root instanceof HTMLElement)) {
    throw new Error("Missing result groups root");
  }
  const activeGroups = Array.from(root.querySelectorAll("[data-result-block]"))
    .filter((group) => group.querySelector("[data-active-task='true']"));
  const activeBlockIds = activeGroups.map((group) => {
    const form = Array.from(group.querySelectorAll("[data-regenerate-form]"))
      .find((item) => item.querySelector("button:disabled"));
    return form?.dataset.blockId;
  }).filter(Boolean);
  await Promise.all(activeBlockIds.map((blockId) => refreshResultBlock(blockId)));
}

async function refreshResultGroups() {
  if (resultGroupsRequest) {
    return resultGroupsRequest;
  }
  window.clearTimeout(resultGroupsTimer);
  resultGroupsTimer = null;
  resultGroupsRefreshNeeded = true;
  resultGroupsRequest = (async () => {
    while (true) {
      const root = document.querySelector("[data-result-groups]");
      if (!(root instanceof HTMLElement) || !root.dataset.resultUrl) {
        throw new Error("Missing result groups root");
      }
      const generation = resultGroupsGeneration;
      const response = await fetch(root.dataset.resultUrl, {
        method: "GET",
        headers: {
          "Accept": "text/html",
          "X-Linguary-Fragment": "result-groups",
        },
        credentials: "same-origin",
      });
      if (generation !== resultGroupsGeneration) {
        await response.body?.cancel();
        continue;
      }
      if (!response.ok) {
        throw new Error(`Request failed: ${response.status}`);
      }
      const markup = await response.text();
      if (generation !== resultGroupsGeneration) {
        continue;
      }
      if (!root.isConnected || document.querySelector("[data-result-groups]") !== root) {
        throw new Error("Result groups root was detached");
      }
      const template = document.createElement("template");
      template.innerHTML = markup.trim();
      const replacement = template.content.firstElementChild;
      if (
        !(replacement instanceof HTMLElement) ||
        template.content.children.length !== 1 ||
        template.content.querySelectorAll("[data-result-groups]").length !== 1 ||
        !replacement.matches("[data-result-groups]") ||
        replacement.dataset.resultUrl !== root.dataset.resultUrl
      ) {
        throw new Error("Invalid result groups response");
      }
      const scrollX = window.scrollX;
      const scrollY = window.scrollY;
      const focus = captureResultFocus(root) || (
        document.activeElement === document.body ? resultGroupsPendingFocus : null
      );
      const nextBlocks = new Map(Array.from(
        replacement.querySelectorAll("[data-result-block]"), (block) => [block.id, block],
      ));
      for (const block of root.querySelectorAll("[data-result-block]")) {
        const nextBlock = nextBlocks.get(block.id);
        const guides = nextBlock?.querySelectorAll("details.ipa-guide");
        block.querySelectorAll("details.ipa-guide").forEach((guide, index) => {
          if (guides?.[index]) {
            guides[index].open = guide.open;
          }
        });
      }
      root.replaceWith(replacement);
      restoreResultFocus(replacement, focus);
      resultGroupsPendingFocus = null;
      window.scrollTo(scrollX, scrollY);
      resultGroupsRefreshNeeded = false;
      resultGroupsRetryDelay = 1500;
      return replacement;
    }
  })().catch((error) => {
    resultGroupsRetryDelay = 2000;
    throw error;
  }).finally(() => {
    resultGroupsRequest = null;
    pollResultGroups(resultGroupsAttempt);
  });
  return resultGroupsRequest;
}

function pollResultGroups(attempt = 1) {
  resultGroupsAttempt = attempt;
  const root = document.querySelector("[data-result-groups]");
  const hasActiveTasks = root instanceof HTMLElement &&
    Boolean(root.querySelector("[data-active-task='true']"));
  if (hasActiveTasks) {
    resultGroupsRefreshNeeded = true;
  }
  if (
    attempt > 120 || resultGroupsTimer !== null || resultGroupsRequest ||
    !(root instanceof HTMLElement) ||
    (!resultGroupsRefreshNeeded && !hasActiveTasks)
  ) {
    return;
  }
  resultGroupsTimer = window.setTimeout(async () => {
    resultGroupsTimer = null;
    resultGroupsAttempt = attempt + 1;
    try {
      await refreshActiveResultBlocks();
      const currentRoot = document.querySelector("[data-result-groups]");
      if (
        resultGroupsRefreshNeeded && currentRoot instanceof HTMLElement &&
        !currentRoot.querySelector("[data-active-task='true']")
      ) {
        await refreshResultGroups();
      }
      await refreshResultHeader();
    } catch (_error) {
      resultGroupsRetryDelay = 2000;
    } finally {
      pollResultGroups(resultGroupsAttempt);
    }
  }, resultGroupsRetryDelay);
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
    proxy_url: getVal("proxy_mode") === "custom" ? getVal("proxy_url") : undefined,
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
  pollResultGroups();
});
