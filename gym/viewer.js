function createViewer({ dayTitle, moves, containerId = 'app' }) {
  const el = document.getElementById(containerId);
  if (!el) return;

  let idx = 0;
  let slideTimer = null;

  function clearTimer() {
    if (slideTimer) {
      clearInterval(slideTimer);
      slideTimer = null;
    }
  }

  function renderMedia(media) {
    const wrap = document.createElement('div');
    wrap.className = 'media-wrap';
    if (!media || !media.type) {
      wrap.innerHTML = '<div style="padding:16px; text-align:center; color:#9ca3af;">No media provided for this move.</div>';
      return wrap;
    }
    if (media.type === 'video' || media.type === 'gif') {
      const v = document.createElement(media.type === 'video' ? 'video' : 'img');
      if (media.type === 'video') {
        v.src = media.src;
        v.autoplay = true;
        v.loop = true;
        v.muted = true;
        v.playsInline = true;
      } else {
        v.src = media.src;
        v.alt = media.alt || '';
      }
      wrap.appendChild(v);
      return wrap;
    }
    if (media.type === 'slideshow') {
      const images = media.images || [];
      if (!images.length) {
        wrap.innerHTML = '<div style="padding:16px; text-align:center; color:#9ca3af;">No images for slideshow.</div>';
        return wrap;
      }
      const img = document.createElement('img');
      let sIdx = 0;
      img.src = images[0];
      wrap.appendChild(img);
      clearTimer();
      slideTimer = setInterval(() => {
        sIdx = (sIdx + 1) % images.length;
        img.src = images[sIdx];
      }, 3000);
      return wrap;
    }
    wrap.innerHTML = '<div style="padding:16px; text-align:center; color:#9ca3af;">Unsupported media type.</div>';
    return wrap;
  }

  function render() {
    clearTimer();
    const move = moves[idx];
    el.innerHTML = '';

    const back = document.createElement('a');
    back.className = 'back';
    back.href = 'index.html';
    back.textContent = '← Back to overview';
    el.appendChild(back);

    const header = document.createElement('div');
    header.className = 'viewer-header';
    header.innerHTML = `<h1>${dayTitle}</h1><div>${idx + 1} / ${moves.length}</div>`;
    el.appendChild(header);

    const sub = document.createElement('p');
    sub.className = 'viewer-sub';
    sub.textContent = 'Use Next/Prev to browse moves.';
    el.appendChild(sub);

    const mediaEl = renderMedia(move.media);
    el.appendChild(mediaEl);

    const t = document.createElement('div');
    t.className = 'move-title';
    t.textContent = move.title;
    el.appendChild(t);

    const meta = document.createElement('p');
    meta.className = 'move-meta';
    meta.textContent = move.meta || '';
    el.appendChild(meta);

    const controls = document.createElement('div');
    controls.className = 'controls';
    const prev = document.createElement('button');
    prev.textContent = '◀ Prev';
    prev.onclick = () => { idx = (idx - 1 + moves.length) % moves.length; render(); };
    const next = document.createElement('button');
    next.textContent = 'Next ▶';
    next.onclick = () => { idx = (idx + 1) % moves.length; render(); };
    controls.appendChild(prev);
    controls.appendChild(next);
    el.appendChild(controls);

    const legend = document.createElement('div');
    legend.className = 'legend';
    legend.textContent = 'Slideshow auto-advances every 3 seconds if a GIF/video is not available.';
    el.appendChild(legend);
  }

  render();
}
