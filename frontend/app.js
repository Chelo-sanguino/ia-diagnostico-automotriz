// Cliente web: consume todos los endpoints del servicio de IA.
// Para servir el front en otro dominio use:  index.html?api=https://mi-servicio.com
const API = (new URLSearchParams(location.search).get('api') || '') + '/api/v1';
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const esc = (v) => String(v ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const pct = (v, d = 1) => (v * 100).toFixed(d) + '%';
const fmtDate = (iso) => new Date(iso).toLocaleString('es-BO', { dateStyle: 'short', timeStyle: 'short' });

async function api(path, opts = {}) {
  const res = await fetch(API + path, opts);
  let data = null;
  try { data = await res.json(); } catch { /* sin cuerpo */ }
  if (!res.ok) {
    const d = data && data.detail;
    const msg = Array.isArray(d) ? d.map(e => `${(e.loc || []).slice(1).join('.')}: ${e.msg}`).join('; ') : (d || res.statusText);
    throw new Error(msg);
  }
  return data;
}

let toastTimer;
function toast(msg) {
  const t = $('#toast'); t.textContent = msg; t.classList.add('show');
  clearTimeout(toastTimer); toastTimer = setTimeout(() => t.classList.remove('show'), 3200);
}
function setMsg(el, text, kind) { el.textContent = text; el.className = 'msg ' + (kind || ''); }

// ------------------------------------------------------------------ tabs --
const loaders = { history: loadHistory, metrics: loadMetrics, train: loadFiles, models: loadModels };
$$('.tab').forEach(btn => btn.addEventListener('click', () => {
  $$('.tab').forEach(b => b.classList.toggle('active', b === btn));
  $$('.panel').forEach(p => p.classList.toggle('active', p.id === 'tab-' + btn.dataset.tab));
  (loaders[btn.dataset.tab] || (() => {}))();
}));

async function refreshPill() {
  try {
    const m = await api('/metrics');
    $('#model-pill').textContent = `Modelo v${m.model_version} · F1 ${m.f1_macro.toFixed(2)}`;
  } catch { $('#model-pill').textContent = 'sin modelo activo'; }
}

// ------------------------------------------------------------ diagnóstico --
$('#predict-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const body = Object.fromEntries(new FormData(e.target).entries());
  const btn = e.submitter; btn.disabled = true;
  try {
    const r = await api('/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    $('#result').className = '';
    $('#result').innerHTML = `
      <div class="conf">Sistema afectado más probable</div>
      <div class="big-cat">${esc(r.predicted_category)}</div>
      <div class="conf">Confianza ${pct(r.confidence)} · modelo v${r.model_version} · registro #${r.id}</div>
      <div class="bar"><span style="width:${r.confidence * 100}%"></span></div>
      <h3>Top 3</h3>
      ${r.top_predictions.map(t => `<div class="top-row"><span>${esc(t.category)}</span>
        <div class="bar blue"><span style="width:${t.probability * 100}%"></span></div><span class="p">${pct(t.probability, 0)}</span></div>`).join('')}`;
  } catch (err) { toast('Error: ' + err.message); }
  finally { btn.disabled = false; }
});

$('#btn-example').addEventListener('click', async () => {
  try {
    const [ex] = await api('/examples?n=1');
    const f = $('#predict-form');
    for (const k of ['symptoms', 'step_1', 'result_1', 'step_2', 'result_2']) f.elements[k].value = ex[k];
    toast(`Ejemplo cargado (categoría real: ${ex.category})`);
  } catch (err) { toast('Error: ' + err.message); }
});

// -------------------------------------------------------------- historial --
const hist = { offset: 0, limit: 10, total: 0 };
async function loadHistory() {
  const cat = $('#hist-filter').value;
  const q = new URLSearchParams({ limit: hist.limit, offset: hist.offset });
  if (cat) q.set('category', cat);
  try {
    const d = await api('/predictions?' + q);
    hist.total = d.total;
    $('#hist-table tbody').innerHTML = d.items.length ? d.items.map(p => `<tr>
      <td>${p.id}</td><td>${fmtDate(p.created_at)}</td><td>${esc(p.symptoms)}</td>
      <td>${esc(p.step_1)}: <b>${esc(p.result_1)}</b><br>${esc(p.step_2)}: <b>${esc(p.result_2)}</b></td>
      <td><span class="tag on">${esc(p.predicted_category)}</span></td><td class="num">${pct(p.confidence, 0)}</td><td>v${p.model_version_id}</td></tr>`).join('')
      : '<tr><td colspan="7" class="empty">Aún no hay predicciones.</td></tr>';
    const to = Math.min(hist.offset + hist.limit, hist.total);
    $('#hist-info').textContent = hist.total ? `${hist.offset + 1}–${to} de ${hist.total}` : '0 resultados';
    $('#hist-prev').disabled = hist.offset === 0;
    $('#hist-next').disabled = hist.offset + hist.limit >= hist.total;
  } catch (err) { toast('Error: ' + err.message); }
}
$('#hist-refresh').addEventListener('click', () => { hist.offset = 0; loadHistory(); });
$('#hist-filter').addEventListener('change', () => { hist.offset = 0; loadHistory(); });
$('#hist-prev').addEventListener('click', () => { hist.offset = Math.max(0, hist.offset - hist.limit); loadHistory(); });
$('#hist-next').addEventListener('click', () => { hist.offset += hist.limit; loadHistory(); });

// --------------------------------------------------------------- métricas --
async function loadMetrics() {
  const box = $('#metrics-content');
  try {
    const m = await api('/metrics');
    const sel = $('#hist-filter');
    if (sel.options.length === 1) m.classes.forEach(c => sel.add(new Option(c, c)));
    const cv = m.nested_cv;
    const rows = m.classes.map(c => ({ c, ...m.per_class[c] }));
    box.innerHTML = `
      <div class="kpis">
        <div class="kpi"><div class="l">Accuracy</div><div class="v">${pct(m.accuracy)}</div><div class="s">baseline clase mayoritaria: ${pct(m.majority_baseline_accuracy ?? 0)}</div></div>
        <div class="kpi"><div class="l">F1-score macro</div><div class="v">${m.f1_macro.toFixed(3)}</div><div class="s">promedio sobre todas las clases</div></div>
        <div class="kpi"><div class="l">Balanced accuracy</div><div class="v">${pct(m.balanced_accuracy)}</div><div class="s">recall promedio por clase</div></div>
        <div class="kpi"><div class="l">Muestras</div><div class="v">${m.training_samples}</div><div class="s">test: ${m.test_samples} · v${m.model_version}</div></div>
      </div>
      ${cv ? `<div class="note"><b>Variabilidad real (CV agrupada anidada, ${cv.folds} particiones):</b>
        F1 macro ${cv.f1_macro_mean.toFixed(3)} ± ${cv.f1_macro_std.toFixed(3)} · Accuracy ${pct(cv.accuracy_mean)} ± ${pct(cv.accuracy_std)}.
        Por partición: ${cv.f1_macro_per_fold.map(v => v.toFixed(2)).join(' / ')}.
        Se evalúa siempre sobre <i>componentes no vistos</i>, por eso el valor del hold-out depende de qué componentes caen en test.</div>` : ''}
      <div class="card">
        <h2>Rendimiento por clase</h2>
        <div class="table-wrap"><table><thead><tr><th>Categoría</th><th class="num">Precision</th><th class="num">Recall</th><th>F1</th><th class="num">Soporte</th></tr></thead><tbody>
        ${rows.map(r => `<tr><td>${esc(r.c)}</td><td class="num">${r.precision.toFixed(2)}</td><td class="num">${r.recall.toFixed(2)}</td>
          <td style="min-width:160px"><div class="top-row" style="grid-template-columns:1fr 40px;margin:0"><div class="bar"><span style="width:${r['f1-score'] * 100}%"></span></div><span class="p">${r['f1-score'].toFixed(2)}</span></div></td>
          <td class="num">${r.support}</td></tr>`).join('')}
        </tbody></table></div>
      </div>
      <div class="card" style="margin-top:18px">
        <h2>Matriz de confusión (hold-out)</h2>
        <p class="hint">Filas = clase real · columnas = clase predicha. La diagonal son los aciertos.</p>
        <div class="table-wrap" id="cm-wrap"></div>
      </div>`;
    renderConfusion(m.confusion_matrix);
  } catch (err) { box.innerHTML = `<div class="card msg err">${esc(err.message)}</div>`; }
}

function renderConfusion({ labels, matrix }) {
  const n = labels.length;
  const short = (s) => s.length > 13 ? s.slice(0, 12) + '…' : s;
  let h = `<div class="cm" style="grid-template-columns:150px repeat(${n},1fr)"><div class="h"></div>`;
  labels.forEach(l => h += `<div class="h" title="${esc(l)}">${esc(short(l))}</div>`);
  matrix.forEach((row, i) => {
    const tot = row.reduce((a, b) => a + b, 0) || 1;
    h += `<div class="rl" title="${esc(labels[i])}">${esc(labels[i])}</div>`;
    row.forEach((v, j) => {
      const a = v / tot;
      const bg = v === 0 ? 'transparent' : (i === j ? `rgba(62,207,142,${.15 + a * .75})` : `rgba(255,107,107,${.15 + a * .75})`);
      h += `<div style="background:${bg}" title="real: ${esc(labels[i])} → pred: ${esc(labels[j])}">${v || ''}</div>`;
    });
  });
  $('#cm-wrap').innerHTML = h + '</div>';
}

// ---------------------------------------------------------- entrenamiento --
async function loadFiles() {
  try {
    const files = await api('/training-files');
    $('#files-table tbody').innerHTML = files.length ? files.map(f => `<tr><td>${f.id}</td><td>${esc(f.original_name)}</td>
      <td>${esc(f.file_format)}</td><td class="num">${f.rows_count}</td><td class="num">${f.categories_count}</td><td>${fmtDate(f.uploaded_at)}</td></tr>`).join('')
      : '<tr><td colspan="6" class="empty">Sin archivos subidos todavía.</td></tr>';
  } catch (err) { toast('Error: ' + err.message); }
}

$('#upload-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const file = $('#file-input').files[0];
  const msg = $('#upload-msg');
  if (!file) return;
  const fd = new FormData(); fd.append('file', file);
  e.submitter.disabled = true; setMsg(msg, 'Validando…');
  try {
    const r = await api('/training-files', { method: 'POST', body: fd });
    setMsg(msg, `✔ "${r.original_name}" válido: ${r.rows_count} filas, ${r.categories_count} categorías.`, 'ok');
    e.target.reset(); loadFiles();
  } catch (err) { setMsg(msg, '✖ ' + err.message, 'err'); }
  finally { e.submitter.disabled = false; }
});

$('#btn-retrain').addEventListener('click', async (e) => {
  const btn = e.currentTarget, msg = $('#retrain-msg'), out = $('#retrain-result');
  btn.disabled = true; out.innerHTML = ''; setMsg(msg, 'Entrenando… (búsqueda de hiperparámetros + CV agrupada)');
  try {
    const r = await api('/retrain', { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ include_uploaded: $('#opt-uploaded').checked, activate: $('#opt-activate').checked }) });
    setMsg(msg, '✔ ' + r.message, 'ok');
    const n = r.new_model, p = r.previous_active_model;
    const d = (a, b) => { const x = a - b; return `<span class="delta ${x >= 0 ? 'up' : 'down'}">${x >= 0 ? '▲' : '▼'} ${Math.abs(x).toFixed(3)}</span>`; };
    out.innerHTML = `<h3>Comparación</h3><div class="table-wrap"><table><thead><tr><th></th><th class="num">Anterior${p ? ' v' + p.id : ''}</th><th class="num">Nueva v${n.id}</th><th class="num">Δ</th></tr></thead><tbody>
      ${[['Accuracy', 'accuracy'], ['F1 macro', 'f1_macro'], ['Balanced acc.', 'balanced_accuracy']].map(([l, k]) =>
        `<tr><td>${l}</td><td class="num">${p ? p[k].toFixed(3) : '—'}</td><td class="num">${n[k].toFixed(3)}</td><td class="num">${p ? d(n[k], p[k]) : ''}</td></tr>`).join('')}
      <tr><td>Muestras</td><td class="num">${p ? p.training_samples : '—'}</td><td class="num">${n.training_samples}</td><td></td></tr></tbody></table></div>
      <p class="hint">Los hold-outs pueden contener componentes distintos; compara también la CV anidada en la pestaña Métricas.</p>`;
    refreshPill();
  } catch (err) { setMsg(msg, '✖ ' + err.message, 'err'); }
  finally { btn.disabled = false; }
});

// --------------------------------------------------------------- versiones --
async function loadModels() {
  try {
    const ms = await api('/models');
    $('#models-table tbody').innerHTML = ms.map(m => `<tr>
      <td>v${m.id} ${m.is_active ? '<span class="tag on">activa</span>' : ''}</td><td>${esc(m.source)}</td>
      <td class="num">${m.training_samples}</td><td class="num">${m.accuracy.toFixed(3)}</td><td class="num">${m.f1_macro.toFixed(3)}</td>
      <td class="num">${m.balanced_accuracy.toFixed(3)}</td><td>${fmtDate(m.created_at)}</td>
      <td>${m.is_active ? '' : `<button class="btn ghost" data-act="${m.id}">Activar</button>`}</td></tr>`).join('');
    $$('[data-act]').forEach(b => b.addEventListener('click', async () => {
      try { await api(`/models/${b.dataset.act}/activate`, { method: 'POST' }); toast(`Versión v${b.dataset.act} activada`); loadModels(); refreshPill(); }
      catch (err) { toast('Error: ' + err.message); }
    }));
  } catch (err) { toast('Error: ' + err.message); }
}

refreshPill();
loadMetrics(); // precarga las clases para el filtro del historial
