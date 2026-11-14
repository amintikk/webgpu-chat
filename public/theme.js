// Applies the saved theme before the page paints (no flash of the wrong theme).
try {
  var t = JSON.parse(localStorage.getItem('webgpu-chat-settings') || '{}').theme;
  if (t === 'dark' || t === 'light') document.documentElement.dataset.theme = t;
} catch (e) {}
