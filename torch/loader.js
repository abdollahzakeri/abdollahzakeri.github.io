// Dynamic loader for hierarchical examples
// Directory convention:
// examples/<categoryId>/<topicId>/<exampleId>.js exporting window.registerExample({...})

(function(){
  const root = 'examples';

  async function loadManifest() {
    try {
      const res = await fetch(`${root}/manifest.json`, { cache: 'no-cache' });
      if (!res.ok) throw new Error('manifest fetch error');
      return await res.json();
    } catch (e) {
      console.warn('manifest.json fetch failed; trying manifest.js fallback…', e);
      // Try to load a JS manifest that sets window.EXAMPLES_MANIFEST
      await new Promise((resolve, reject) => {
        const s = document.createElement('script');
        s.src = `${root}/manifest.js`;
        s.async = false;
        s.onload = resolve;
        s.onerror = reject;
        document.head.appendChild(s);
      }).catch((err) => {
        console.warn('manifest.js load failed; falling back to embedded data.js if present.', err);
      });
      if (window.EXAMPLES_MANIFEST && window.EXAMPLES_MANIFEST.files) {
        return window.EXAMPLES_MANIFEST;
      }
      // If legacy data.js exists, keep using it.
      if (window.PYTORCH_COOKBOOK) {
        document.dispatchEvent(new Event('examples-ready'));
      }
      return null;
    }
  }

  function ensureCookbook() {
    if (!window.PYTORCH_COOKBOOK) {
      window.PYTORCH_COOKBOOK = { categories: [] };
    }
  }

  function upsertCategory(cat) {
    ensureCookbook();
    const existing = window.PYTORCH_COOKBOOK.categories.find(c => c.id === cat.id);
    if (!existing) {
      window.PYTORCH_COOKBOOK.categories.push({ id: cat.id, name: cat.name, summary: cat.summary, topics: [] });
    }
  }

  function upsertTopic(categoryId, topic) {
    const cat = window.PYTORCH_COOKBOOK.categories.find(c => c.id === categoryId);
    if (!cat) return;
    const existing = cat.topics.find(t => t.id === topic.id);
    if (!existing) cat.topics.push(topic);
  }

  // Called by generated example files
  window.registerExample = function(categoryId, topicInfo, example) {
    upsertCategory({ id: categoryId, name: topicInfo.categoryName, summary: topicInfo.categorySummary });
    upsertTopic(categoryId, {
      id: topicInfo.topicId,
      name: topicInfo.topicName,
      tags: example.tags || [],
      description: example.description || '',
      meta: example.meta || '',
      code: example.code || ''
    });
  };

  (async function init(){
    const manifest = await loadManifest();
    if (!manifest) return; // fallback handled
    // Load all example scripts sequentially to ensure registration order
    for (const entry of manifest.files) {
      const path = `${root}/${entry}`;
      try {
        await new Promise((resolve, reject) => {
          const s = document.createElement('script');
          s.src = path; s.async = false;
          s.onload = resolve; s.onerror = reject;
          document.head.appendChild(s);
        });
      } catch (e) {
        console.error('Failed to load example', path, e);
      }
    }
    document.dispatchEvent(new Event('examples-ready'));
  })();
})();


