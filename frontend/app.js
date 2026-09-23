/* Состояние интерфейса: чат, индикаторы, панели навыков и метрик. */

import * as api from './api.js';
import { playPlan, setFace, onBusyChange } from './robot.js';
import { refreshSkills, refreshProposals, onLearningChange } from './skills.js';

const STATE_POLL_MS = 15000;

const chatEl = document.getElementById('chat');
const formEl = document.getElementById('composer');
const inputEl = document.getElementById('input');
const sendEl = document.getElementById('send');
const quickEl = document.getElementById('quick');
const statsEl = document.getElementById('stats');
const metricsEl = document.getElementById('metrics');

const quickButtons = [...quickEl.querySelectorAll('button')];

const ENGINE_LABELS = { laya: 'Laya', gemini: 'Gemini', button: 'Кнопка' };
const ENGINE_CLASSES = { laya: 'badge-laya', gemini: 'badge-gemini', button: 'badge-button' };

/* ─── Индикаторы ────────────────────────────────────────────────────────── */

function levelClass(value) {
  if (value < 25) return 'lvl-bad';
  if (value < 50) return 'lvl-warn';
  return '';
}

function renderState(state) {
  if (!state) return;
  for (const key of ['mood', 'energy', 'fullness']) {
    const value = Math.max(0, Math.min(100, Number(state[key]) || 0));
    const bar = statsEl.querySelector(`[data-bar="${key}"]`);
    const label = statsEl.querySelector(`[data-val="${key}"]`);
    bar.style.width = `${value}%`;
    bar.className = levelClass(value);
    label.textContent = Math.round(value);
  }
  if (state.face) setFace(state.face);
}

/* ─── Чат ───────────────────────────────────────────────────────────────── */

function scrollChat() {
  chatEl.scrollTop = chatEl.scrollHeight;
}

function addMessage(kind, text) {
  const wrap = document.createElement('div');
  wrap.className = `msg msg-${kind}`;
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.textContent = text;
  wrap.append(bubble);
  chatEl.append(wrap);
  scrollChat();
  return wrap;
}

function engineBadge(engine) {
  const badge = document.createElement('span');
  const key = String(engine ?? '');
  badge.className = `badge ${ENGINE_CLASSES[key] ?? ''}`.trim();
  badge.textContent = ENGINE_LABELS[key] ?? (key || 'неизвестно');
  return badge;
}

function voteButtons(interactionId) {
  const box = document.createElement('div');
  box.className = 'vote';

  const send = async (button, value) => {
    [...box.children].forEach((b) => { b.disabled = true; });
    button.classList.add('picked');
    try {
      await api.feedback(interactionId, value);
    } catch (error) {
      console.error(error);
      button.classList.remove('picked');
      [...box.children].forEach((b) => { b.disabled = false; });
    }
  };

  for (const [label, value, title] of [['👍', 1, 'Хорошая реакция'], ['👎', -1, 'Плохая реакция']]) {
    const button = document.createElement('button');
    button.type = 'button';
    button.textContent = label;
    button.title = title;
    button.addEventListener('click', () => send(button, value));
    box.append(button);
  }
  return box;
}

function addReply(reply) {
  const wrap = addMessage('bot', reply.reply ?? '');

  const meta = document.createElement('div');
  meta.className = 'meta';
  meta.append(engineBadge(reply.engine));

  if (Number.isFinite(reply.latency_ms)) {
    const latency = document.createElement('span');
    latency.textContent = `${reply.latency_ms} мс`;
    meta.append(latency);
  }
  if (reply.confidence !== null && reply.confidence !== undefined) {
    const confidence = document.createElement('span');
    confidence.textContent = `уверенность ${Math.round(reply.confidence * 100)}%`;
    meta.append(confidence);
  }
  if (reply.interaction_id !== null && reply.interaction_id !== undefined) {
    meta.append(voteButtons(reply.interaction_id));
  }

  wrap.append(meta);
  scrollChat();
}

/* ─── Отправка команд ───────────────────────────────────────────────────── */

let sending = false;

function setControlsEnabled(enabled) {
  quickButtons.forEach((button) => { button.disabled = !enabled; });
  sendEl.disabled = !enabled;
  inputEl.disabled = !enabled;
}

onBusyChange((busy) => setControlsEnabled(!busy && !sending));

async function send(call) {
  if (sending) return;
  sending = true;
  setControlsEnabled(false);
  try {
    const reply = await call();
    addReply(reply);
    renderState(reply.state);
    await playPlan(reply.actions ?? []);
    /* Новый кластер для майнера появляется только на ответе учителя, так что
       предложения перезапрашиваются именно тогда; статистика — каждый раз. */
    refreshSkills();
    refreshMetrics();
    if (reply.engine === 'gemini') refreshProposals();
  } catch (error) {
    console.error(error);
    addMessage('system', 'Не получилось связаться с Pixel. Попробуй ещё раз.');
  } finally {
    sending = false;
    setControlsEnabled(true);
  }
}

quickEl.addEventListener('click', (event) => {
  const button = event.target.closest('button[data-action]');
  if (!button) return;
  addMessage('user', button.textContent);
  send(() => api.action(button.dataset.action));
});

formEl.addEventListener('submit', (event) => {
  event.preventDefault();
  const text = inputEl.value.trim();
  if (!text) return;
  inputEl.value = '';
  addMessage('user', text);
  send(() => api.chat(text));
});

/* ─── Метрики ───────────────────────────────────────────────────────────── */

function metricCard(label, value, lead = false) {
  const el = document.createElement('div');
  el.className = lead ? 'metric metric-lead' : 'metric';
  const caption = document.createElement('span');
  caption.textContent = label;
  const strong = document.createElement('b');
  strong.textContent = value ?? 'нет данных';
  if (value === null) strong.className = 'muted';
  el.append(caption, strong);
  return el;
}

/* null -> карточка покажет «нет данных» мелким шрифтом */
const ms = (value) => (Number.isFinite(value) ? `${Math.round(value)} мс` : null);

function renderMetrics(metrics) {
  const share = Number.isFinite(metrics.laya_share) ? `${Math.round(metrics.laya_share * 100)}%` : null;
  metricsEl.replaceChildren(
    metricCard('Доля команд без Gemini', share, true),
    metricCard('Laya, среднее время', ms(metrics.avg_latency_laya_ms)),
    metricCard('Gemini, среднее время', ms(metrics.avg_latency_gemini_ms)),
    metricCard('Активных навыков', String(metrics.skills_active ?? 0)),
    metricCard('Всего команд', String(metrics.total_commands ?? 0)),
  );
}

async function refreshMetrics() {
  try {
    renderMetrics(await api.getMetrics() ?? {});
  } catch (error) {
    console.error(error);
  }
}

/* accept / reject / «поискать новые навыки» меняют и метрики тоже. */
onLearningChange(refreshMetrics);

/* ─── Старт ─────────────────────────────────────────────────────────────── */

async function pollState() {
  try {
    renderState(await api.getState());
  } catch (error) {
    console.error(error);
  }
}

addMessage('system', 'Pixel проснулся. Напиши ему или нажми кнопку.');
pollState();
refreshSkills();
refreshProposals();
refreshMetrics();
setInterval(pollState, STATE_POLL_MS);

/* Ручная проверка выражений и анимаций из консоли. */
window.pixel = { setFace, playPlan };

const brokenProbe = {
