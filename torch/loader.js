// Dynamic loader for hierarchical examples
// Directory convention:
// examples/<categoryId>/<topicId>/<exampleId>.js exporting window.registerExample({...})

(function(){
  const root = 'examples';

  async function loadManifest() {
    // Always rely on manifest.js (preloaded in index.html) for GitHub Pages CSP compatibility
    if (window.EXAMPLES_MANIFEST) return window.EXAMPLES_MANIFEST;
    // Fallback: attempt to load it if not included
    try {
      await new Promise((resolve, reject) => {
        const s = document.createElement('script');
        s.src = `${root}/manifest.js`;
        s.async = false;
        s.onload = resolve;
        s.onerror = reject;
        document.head.appendChild(s);
      });
      return window.EXAMPLES_MANIFEST || null;
    } catch (e) {
      console.warn('manifest.js load failed; falling back to embedded data.js if present.', e);
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

  // CSP-safe parser for our example JS files. Avoids eval and Function.
  function parseExampleText(text) {
    try {
      const catMatch = text.match(/window\.registerExample\(\s*'([^']+)'/);
      if (!catMatch) return null;
      const categoryId = catMatch[1];
      const get = (key) => {
        const m = text.match(new RegExp(key + ":\\s*'([^']*)'"));
        return m ? m[1] : '';
      };
      const topicId = get('topicId');
      const topicName = get('topicName');
      const categoryName = get('categoryName');
      const categorySummary = get('categorySummary');
      const id = get('id');
      const name = get('name');
      const meta = get('meta');
      const description = get('description');
      // tags: ['a','b']
      const tagsMatch = text.match(/tags:\s*\[([^\]]*)\]/);
      const tags = tagsMatch ? tagsMatch[1].split(',').map(s => s.trim().replace(/^'|'$/g,'')).filter(Boolean) : [];
      // code: `...`
      const codeMatch = text.match(/code:\s*`([\s\S]*?)`/);
      const code = codeMatch ? codeMatch[1] : '';
      return {
        categoryId,
        topicInfo: { categoryName, categorySummary, topicId, topicName },
        example: { id: id || topicId, name: name || topicName, tags, meta, description, code },
      };
    } catch (e) {
      return null;
    }
  }

  (async function init(){
    const manifest = await loadManifest();
    if (!manifest) return; // fallback handled
    // Load example JS files via script tags (CSP-friendly, supports file:// and GitHub Pages)
    const fileList = manifest.files || [];
    let loadedCount = 0;
    await Promise.all(fileList.map((entry) => new Promise((resolve) => {
      const path = `${root}/${entry}`;
      const s = document.createElement('script');
      s.src = path; s.async = false;
      s.onload = () => { loadedCount += 1; resolve(); };
      s.onerror = () => { console.error('Failed to load example', path); resolve(); };
      document.head.appendChild(s);
    })));
    if (loadedCount > 0) document.dispatchEvent(new Event('examples-ready'));
    else document.dispatchEvent(new Event('examples-ready'));
  })();
})();


