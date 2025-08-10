// Minimal client-side router + rendering
// Data lives in data.js as global "PYTORCH_COOKBOOK" with categories, topics, and examples

(function () {
  const views = {
    home: document.getElementById('view-home'),
    category: document.getElementById('view-category'),
    example: document.getElementById('view-example'),
  };
  const grids = {
    home: document.getElementById('home-grid'),
    category: document.getElementById('category-grid'),
  };
  const els = {
    search: document.getElementById('search'),
    backBtn: document.getElementById('backBtn'),
    categoryTitle: document.getElementById('category-title'),
    exampleTitle: document.getElementById('example-title'),
    exampleMeta: document.getElementById('example-meta'),
    exampleDesc: document.getElementById('example-desc'),
    codeBlock: document.getElementById('code-block'),
    copyBtn: document.getElementById('copyBtn'),
  };

  function setView(name) {
    for (const key of Object.keys(views)) {
      views[key].classList.toggle('view-active', key === name);
    }
    els.backBtn.hidden = name === 'home';
  }

  function renderHome(filter = '') {
    setView('home');
    const query = filter.trim().toLowerCase();
    grids.home.innerHTML = '';
    const categories = (window.PYTORCH_COOKBOOK?.categories ?? []).filter(c =>
      !query || c.name.toLowerCase().includes(query)
    );
    for (const category of categories) {
      const div = document.createElement('div');
      div.className = 'card';
      div.innerHTML = `<div class="card-title">${category.name}</div><div class="card-sub">${category.summary ?? ''}</div>`;
      div.onclick = () => navigateToCategory(category.id);
      grids.home.appendChild(div);
    }
  }

  function renderCategory(categoryId, filter = '') {
    const cookbook = window.PYTORCH_COOKBOOK;
    if (!cookbook || !Array.isArray(cookbook.categories)) { setView('home'); return; }
    setView('category');
    const cat = cookbook.categories.find(c => c.id === categoryId);
    if (!cat) { renderHome(); return; }
    els.categoryTitle.textContent = cat.name;
    const query = filter.trim().toLowerCase();
    grids.category.innerHTML = '';
    for (const topic of cat.topics) {
      const matches = !query || topic.name.toLowerCase().includes(query) || (topic.tags ?? []).some(t => t.toLowerCase().includes(query));
      if (!matches) continue;
      const div = document.createElement('div');
      div.className = 'card';
      const sub = topic.tags && topic.tags.length ? `#${topic.tags.slice(0, 3).join(' #')}` : '';
      div.innerHTML = `<div class="card-title">${topic.name}</div><div class="card-sub">${sub}</div>`;
      div.onclick = () => navigateToExample(categoryId, topic.id);
      grids.category.appendChild(div);
    }
  }

  function renderExample(categoryId, topicId) {
    const cookbook = window.PYTORCH_COOKBOOK;
    if (!cookbook || !Array.isArray(cookbook.categories)) { setView('home'); return; }
    setView('example');
    const cat = cookbook.categories.find(c => c.id === categoryId);
    const topic = cat?.topics.find(t => t.id === topicId);
    if (!cat || !topic) { renderHome(); return; }
    els.exampleTitle.textContent = `${cat.name} · ${topic.name}`;
    els.exampleMeta.textContent = topic.meta ?? '';
    els.exampleDesc.textContent = topic.description ?? '';
    els.codeBlock.textContent = topic.code?.trim() ?? '';
    if (window.Prism && typeof Prism.highlightElement === 'function') {
      Prism.highlightElement(els.codeBlock);
    }
    els.copyBtn.onclick = () => {
      navigator.clipboard.writeText(topic.code ?? '').then(() => {
        els.copyBtn.textContent = 'Copied!';
        setTimeout(() => (els.copyBtn.textContent = 'Copy Code'), 1200);
      });
    };
  }

  function navigateToCategory(categoryId) {
    history.pushState({ v: 'category', categoryId }, '', `#/${encodeURIComponent(categoryId)}`);
    renderCategory(categoryId, els.search.value);
  }
  function navigateToExample(categoryId, topicId) {
    history.pushState({ v: 'example', categoryId, topicId }, '', `#/${encodeURIComponent(categoryId)}/${encodeURIComponent(topicId)}`);
    renderExample(categoryId, topicId);
  }

  function parseLocation() {
    const hash = location.hash.replace(/^#\/?/, '');
    if (!hash) return { v: 'home' };
    const parts = hash.split('/').map(decodeURIComponent);
    if (parts.length === 1 && parts[0]) return { v: 'category', categoryId: parts[0] };
    if (parts.length >= 2) return { v: 'example', categoryId: parts[0], topicId: parts.slice(1).join('/') };
    return { v: 'home' };
  }

  function handleRoute() {
    const st = parseLocation();
    const ready = window.PYTORCH_COOKBOOK && Array.isArray(window.PYTORCH_COOKBOOK.categories) && window.PYTORCH_COOKBOOK.categories.length >= 0;
    if (!ready) return; // wait for examples-ready
    if (st.v === 'home') renderHome(els.search.value);
    else if (st.v === 'category') renderCategory(st.categoryId, els.search.value);
    else if (st.v === 'example') renderExample(st.categoryId, st.topicId);
  }

  els.backBtn.addEventListener('click', () => {
    const st = parseLocation();
    if (st.v === 'example') {
      history.back();
    } else if (st.v === 'category') {
      history.pushState({ v: 'home' }, '', '#/');
      renderHome(els.search.value);
    }
  });

  els.search.addEventListener('input', () => handleRoute());
  window.addEventListener('popstate', () => handleRoute());
  window.addEventListener('hashchange', () => handleRoute());
  document.addEventListener('examples-ready', () => handleRoute());

  // Initial render (after examples load)
  document.addEventListener('DOMContentLoaded', () => {
    // Defer to examples-ready to avoid race conditions
  });
})();


