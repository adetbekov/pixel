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

/* Почему учитель не ответил, когда не ответил. `engine` отвечает на «кого
   спросили», и на исчерпанной квоте это по-прежнему `gemini` — значит сказать
   «ответа не было» он не может, и до JEB-1603 квота выглядела в чате ровно как
   «модель не поняла команду». Ключ приходит в `teacher_status` (null на любом
   обычном ответе); по тексту реплики отличать нельзя — первая же правка
   копирайта или локализация сломала бы это молча. */
const STATUS_LABELS = { quota_exhausted: 'квота исчерпана' };
const STATUS_CLASSES = { quota_exhausted: 'badge-quota' };
const STATUS_TITLES = {
  quota_exhausted: 'Учитель не ответил: исчерпана квота Gemini. Это не «робот не понял».',
};

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

/* Вторая плашка рядом с движком, а не вместо него: «спросили Gemini» и «ответа
   не пришло» — два разных факта, и оба нужны. Неизвестный статус не рисуем
   вовсе — сервер старше фронта отдаёт null, сервер новее может добавить ключ,
   которого здесь ещё нет, и «неизвестно» в чате хуже, чем ничего. */
function statusBadge(status) {
  const key = String(status ?? '');
  if (!STATUS_LABELS[key]) return null;
  const badge = document.createElement('span');
  badge.className = `badge ${STATUS_CLASSES[key]}`;
  badge.textContent = STATUS_LABELS[key];
  badge.title = STATUS_TITLES[key];
  return badge;
}

function voteButtons(interactionId, current = null) {
  const box = document.createElement('div');
  box.className = 'vote';

  /* Оценка меняет здоровье навыка на сервере, поэтому после неё перечитываем и
     навыки, и метрики: плохой навык мог именно сейчас отключиться. */
  const send = async (button, value) => {
    [...box.children].forEach((b) => { b.disabled = true; });
    button.classList.add('picked');
    try {
      await api.feedback(interactionId, value);
      refreshSkills();
      refreshMetrics();
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
    /* Оценка из базы: после перезагрузки страницы она видна, а не сброшена. */
    if (current === value) button.classList.add('picked');
    if (current !== null) button.disabled = true;
    button.addEventListener('click', () => send(button, value));
    box.append(button);
  }
  return box;
}

function addReply(reply, vote = null) {
  const wrap = addMessage('bot', reply.reply ?? '');

  const meta = document.createElement('div');
  meta.className = 'meta';
  meta.append(engineBadge(reply.engine));

  const status = statusBadge(reply.teacher_status);
  if (status) meta.append(status);

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
    meta.append(voteButtons(reply.interaction_id, vote));
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

/* null -> карточка покажет «нет данных» мелким шрифтом.
   Ноль тоже «нет данных»: латентности считаются за сутки, и 0 мс означает, что
   за окно такого вызова не было, а не что робот ответил мгновенно. */
const ms = (value) => (Number.isFinite(value) && value > 0 ? `${Math.round(value)} мс` : null);

const percent = (value) => `${Math.round((Number(value) || 0) * 100)}%`;

/* Главная карточка: крупная доля, полоса прогресса и та же доля за сутки.
   Пока команд не было, доля не «0%», а «пока нет данных» — ноль читался бы как
   «Pixel ничему не научился», хотя его просто ещё ни о чём не просили. */
function shareCard(metrics) {
  const total = Number(metrics.total_commands) || 0;
  const el = document.createElement('div');
  el.className = 'metric metric-lead metric-share';

  const caption = document.createElement('span');
  caption.textContent = 'Доля команд без Gemini';
  el.append(caption);

  const strong = document.createElement('b');
  strong.textContent = total ? percent(metrics.laya_share) : 'пока нет данных';
  if (!total) strong.className = 'muted';
  el.append(strong);

  if (total) {
    const bar = document.createElement('div');
    bar.className = 'metric-bar';
    const fill = document.createElement('i');
    fill.style.width = percent(metrics.laya_share);
    bar.append(fill);

    const note = document.createElement('small');
    note.textContent =
      `за всё время · ${percent(metrics.laya_share_24h)} за 24 часа · команд: ${total}`;
    el.append(bar, note);
  }
  return el;
}

function renderMetrics(metrics) {
  const disabled = Number(metrics.skills_disabled) || 0;
  const cards = [
    shareCard(metrics),
    metricCard('Laya, среднее время за 24 часа', ms(metrics.avg_latency_laya_ms)),
    metricCard('Gemini, среднее время за 24 часа', ms(metrics.avg_latency_gemini_ms)),
    metricCard('Активных навыков', String(metrics.skills_active ?? 0)),
  ];
  /* Отключённые показываем, только когда они есть: пустая карточка «0» просто
     занимала бы место в панели. */
  if (disabled) cards.push(metricCard('Отключено навыков', String(disabled)));
  /* То же и для застрявших кластеров: ноль — нормальное состояние, а не новость.
     Ненулевое значение означает, что майнер перестал перерисовывать один и тот
     же набор случаев — раньше это было видно только в логе (JEB-1579). */
  const stuck = Number(metrics.clusters_stuck) || 0;
  if (stuck) cards.push(metricCard('Застрявших кластеров', String(stuck)));
  metricsEl.replaceChildren(...cards);
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

/* Кнопочная интеракция хранится под именем действия (`feed`), а в чат живьём
   попадает подпись кнопки («Покормить»). Восстановленный разговор должен
   совпадать с тем, что пользователь видел, поэтому имя переводится обратно по
   самим кнопкам — второго списка подписей не заводим. */
function userText(item) {
  const text = item.user_text ?? '';
  if (item.engine !== 'button') return text;
  const button = quickButtons.find((candidate) => candidate.dataset.action === text);
  return button ? button.textContent : text;
}

/* Чат восстанавливается из базы: иначе оценка сохранена на сервере, но после
   перезагрузки её не видно — и это читается как «моё 👎 не сохранилось». */
async function restoreHistory() {
  let history = [];
  try {
    history = await api.getHistory();
  } catch (error) {
    console.error(error);
  }
  if (!Array.isArray(history) || history.length === 0) {
    addMessage('system', 'Pixel проснулся. Напиши ему или нажми кнопку.');
    return;
  }
  for (const item of history) {
    addMessage('user', userText(item));
    addReply(item, item.feedback ?? null);
  }
  addMessage('system', 'Pixel проснулся. Выше — прошлый разговор.');
}

restoreHistory();
pollState();
refreshSkills();
refreshProposals();
refreshMetrics();
setInterval(pollState, STATE_POLL_MS);

/* Ручная проверка выражений и анимаций из консоли. */
window.pixel = { setFace, playPlan };
