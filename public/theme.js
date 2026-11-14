// Applies the saved theme before the page paints (no flash of the wrong theme), and enables the
// entrance animations; if the page script never runs, everything is shown after 2.5 s.
try {
  var t = JSON.parse(localStorage.getItem('webgpu-chat-settings') || '{}').theme;
  if (t === 'dark' || t === 'light') document.documentElement.dataset.theme = t;
} catch (e) {}
document.documentElement.classList.add('js');
setTimeout(function () {
  if (!window.siteReady) document.documentElement.classList.remove('js');
}, 2500);
