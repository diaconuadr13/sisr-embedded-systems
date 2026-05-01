// Small helpers; most interactivity is HTMX + Chart.js inline.
document.body.addEventListener('htmx:responseError', (e) => {
  console.warn('HTMX response error', e.detail);
});
