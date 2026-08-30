document.addEventListener("submit", async (event) => {
  const form = event.target;
  if (!(form instanceof HTMLFormElement)) {
    return;
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
