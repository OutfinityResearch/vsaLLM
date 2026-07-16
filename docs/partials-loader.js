(() => {
  const currentFile = window.location.pathname.split('/').pop() || 'index.html';

  async function loadPartial(node) {
    const source = node.dataset.include;
    if (!source) return;
    try {
      const response = await fetch(source, { cache: 'no-cache' });
      if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
      node.innerHTML = await response.text();
      node.dataset.loaded = 'true';
      for (const link of node.querySelectorAll('.primary-nav a')) {
        const target = new URL(link.getAttribute('href'), window.location.href);
        const targetFile = target.pathname.split('/').pop() || 'index.html';
        const active = targetFile === currentFile;
        link.classList.toggle('is-active', active);
        if (active) link.setAttribute('aria-current', 'page');
      }
    } catch (error) {
      node.innerHTML = `<p class="partial-error">Shared documentation fragment could not be loaded: ${String(error.message)}</p>`;
    }
  }

  document.addEventListener('DOMContentLoaded', () => {
    Promise.all([...document.querySelectorAll('[data-include]')].map(loadPartial));
  });
})();
