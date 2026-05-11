const copyButton = document.querySelector("[data-copy-target]");

if (copyButton) {
  copyButton.addEventListener("click", async () => {
    const targetId = copyButton.getAttribute("data-copy-target");
    const target = targetId ? document.getElementById(targetId) : null;

    if (!target) {
      return;
    }

    const originalLabel = copyButton.textContent;

    try {
      await navigator.clipboard.writeText(target.textContent.trim());
      copyButton.textContent = "Copied";
    } catch {
      copyButton.textContent = "Select text";
    }

    window.setTimeout(() => {
      copyButton.textContent = originalLabel;
    }, 1600);
  });
}
