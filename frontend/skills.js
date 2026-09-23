/* Панель обучения: библиотека навыков и карточки предложенных навыков.

   Карточка предложения — главный экран проекта: это тот момент, когда робот
   научился. Поэтому правила навыка разбираются словами, а не показываются
   как JSON, и `match_rate` стоит крупно рядом с заголовком.

   Формы ответов — `SkillCard` и `Proposal` из `backend/api.py`; примеры фраз
   живут в `skill.examples`, отдельного поля у предложения нет. */

import * as api from './api.js';

const MAX_EXAMPLES = 8;
const TOAST_MS = 4000;
const TOAST_FADE_MS = 200;
const DISMISS_MS = 280;

const skillsEl = document.getElementById('skills');
const proposalsEl = document.getElementById('proposals');
const mineEl = document.getElementById('mine');
const toastsEl = document.getElementById('toasts');

const reduced = window.matchMedia('(prefers-reduced-motion: reduce)');

/* ─── Словари перевода ──────────────────────────────────────────────────── */

const ORIGIN_WORDS = { seed: 'стартовый', mined: 'выучен сам' };

/* `disabled_reason` из `SkillCard`; автоматическая причина пока ровно одна. */
const DISABLED_REASONS = { dislike_rate: 'отключён из-за дизлайков' };

/* Каждая шкала склоняется по-своему, поэтому фраза целиком, а не «шкала» + «слово». */
const STATE_PHRASES = {
  mood: { low: 'настроение плохое', mid: 'настроение среднее', high: 'настроение хорошее' },
  energy: { low: 'энергия низкая', mid: 'энергия средняя', high: 'энергия высокая' },
  fullness: { low: 'робот голодный', mid: 'робот наполовину сыт', high: 'робот сытый' },
};

/* Длинный оператор первым, иначе `>=` прочитается как `>` (как в backend/brain/skill.py). */
const SCORE_OPS = [
  ['>=', 'не ниже'],
  ['<=', 'не выше'],
  ['==', 'ровно'],
  ['>', 'выше'],
  ['<', 'ниже'],
];

const ACTION_WORDS = {
  jump: { icon: '⤴', word: 'прыгнуть' },
  dance: { icon: '🎵', word: 'станцевать' },
  sleep: { icon: '💤', word: 'уснуть' },
  eat: { icon: '🍎', word: 'поесть' },
  spin: { icon: '🌀', word: 'покрутиться' },
  wave: { icon: '👋', word: 'помахать' },
  /* Два действия с аргументами — слово ниже дополняется значением аргумента. */
  say: { icon: '💬', word: 'сказать' },
  set_face: { icon: '🙂', word: 'лицо' },
};

const FACE_WORDS = {
  happy: 'радостное',
  sad: 'грустное',
  sleepy: 'сонное',
  angry: 'сердитое',
  curious: 'любопытное',
};

/* ─── Мелкие помощники ──────────────────────────────────────────────────── */

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function plural(count, one, few, many) {
  if (count % 100 >= 11 && count % 100 <= 14) return many;
  if (count % 10 === 1) return one;
  if (count % 10 >= 2 && count % 10 <= 4) return few;
  return many;
}

/* 1.0 -> «100%», 0.833 -> «83%». Дробная часть здесь ничего не значит. */
const percent = (value) => `${Math.round(Number(value) * 100)}%`;

function notice(container, text) {
  container.replaceChildren(el('div', 'empty', text));
}

/* ─── Тосты ─────────────────────────────────────────────────────────────── */

function toast(text) {
  const node = el('div', 'toast', text);
  toastsEl.append(node);
  setTimeout(() => {
    node.classList.add('is-leaving');
    setTimeout(() => node.remove(), TOAST_FADE_MS);
  }, TOAST_MS);
}

/* ─── Список навыков ────────────────────────────────────────────────────── */

function badge(className, text) {
  return el('span', `badge ${className}`, text);
}

function skillCard(skill) {
  const disabled = skill.status === 'disabled';
  const card = el('article', `card skill${disabled ? ' is-off' : ''}`);

  const head = el('div', 'card-head');
  head.append(el('h3', null, skill.name ?? skill.id ?? 'Навык'));
  head.append(badge(
    skill.origin === 'mined' ? 'badge-mined' : 'badge-seed',
    ORIGIN_WORDS[skill.origin] ?? String(skill.origin ?? ''),
  ));
  if (disabled) head.append(badge('badge-off', 'отключён'));

  const uses = Number(skill.uses) || 0;
  const stats = [
    `использован ${uses} ${plural(uses, 'раз', 'раза', 'раз')}`,
    `👍 ${Number(skill.likes) || 0}`,
    `👎 ${Number(skill.dislikes) || 0}`,
  ];
  /* Навык, который просто исчез из ответов, выглядит как поломка. Причина
     отключения пишется прямо на карточке. */
  if (disabled) stats.push(DISABLED_REASONS[skill.disabled_reason] ?? 'отключён');

  card.append(
    head,
    el('p', 'card-note', skill.description ?? ''),
    el('p', 'card-stats', stats.join(' · ')),
  );
  return card;
}

function renderSkills(skills) {
  if (!skills.length) {
    notice(skillsEl, 'Навыков пока нет. Pixel учится на новых командах.');
    return;
  }
  /* Отключённые — вниз, внутри группы по убыванию использований. */
  const ordered = [...skills].sort((a, b) =>
    (a.status === 'disabled') - (b.status === 'disabled')
    || (Number(b.uses) || 0) - (Number(a.uses) || 0));
  skillsEl.replaceChildren(...ordered.map(skillCard));
}

/* ─── Разбор правил навыка словами ──────────────────────────────────────── */

/* `">=1.5"` + ["простой","обычный","эффектный"] -> «не ниже „эффектный"».
   Уровень — индекс в `criteria`; без них остаётся голое число. */
function scoreText(expected, criteria) {
  const op = SCORE_OPS.find(([token]) => expected.startsWith(token));
  const raw = op ? expected.slice(op[0].length) : expected;
  const level = Array.isArray(criteria) ? criteria[Math.round(Number(raw))] : undefined;
  return `${op ? op[1] : 'ровно'} ${level ? `«${level}»` : raw}`;
}

function conditionText(key, expected, questions) {
  const scale = STATE_PHRASES[key];
  if (scale) return scale[expected] ?? `${key}: ${expected}`;

  const question = questions[key];
  /* Неизвестное условие печатаем как есть — карточка важнее полноты словаря. */
  if (!question) return `${key}: ${expected}`;
  const answer = `ответ на «${question.instructions ?? key}» —`;

  if (question.type === 'noul' && (expected === 'yes' || expected === 'no')) {
    return `${answer} ${expected === 'yes' ? 'да' : 'нет'}`;
  }
  if (question.type === 'score') return `${answer} ${scoreText(expected, question.criteria)}`;
  return `${answer} «${expected}»`;
}

function whenText(when, questions) {
  const parts = Object.entries(when ?? {})
    .map(([key, expected]) => conditionText(key, String(expected), questions));
  return parts.length ? `если ${parts.join(' и ')}` : 'в остальных случаях';
}

function actionChip(step) {
  const name = step?.action;
  const args = step?.args ?? {};
  const spec = ACTION_WORDS[name];
  /* После валидации бэкенда сюда не попасть, но ломать карточку из-за этого незачем. */
  if (!spec) return el('span', 'chip', String(name ?? 'неизвестное действие'));

  let text = spec.word;
  if (name === 'say') text = `${spec.word} «${args.text ?? ''}»`;
  if (name === 'set_face') text = `${spec.word} ${FACE_WORDS[args.face] ?? args.face ?? ''}`.trim();

  const chip = el('span', 'chip');
  chip.append(el('i', 'chip-icon', spec.icon), document.createTextNode(text));
  return chip;
}

function ruleRow(rule, questions) {
  const row = el('li', 'rule');
  row.append(el('span', 'rule-when', whenText(rule?.when, questions)));
  const then = el('span', 'rule-then');
  then.append(...(rule?.actions ?? []).map(actionChip));
  row.append(then);
  return row;
}

/* ─── Карточка предложения ──────────────────────────────────────────────── */

function examplesList(examples) {
  const list = el('ul', 'examples');
  list.append(...examples.slice(0, MAX_EXAMPLES).map((text) => el('li', null, String(text))));
  const rest = examples.length - MAX_EXAMPLES;
  if (rest > 0) list.append(el('li', 'examples-more', `и ещё ${rest}`));
  return list;
}

/* `match_rate` — доля команд группы, которые распознаватель отдаёт этому навыку
   (`backend/miner/backtest.py`). Совпадение с планами учителя считается рядом,
   называется `agreement` и в карточку не выводится: это разброс сырья, а не
   качество навыка, и цифра про учителя на карточке «принять навык?» читалась бы
   как оценка навыка. Подпись здесь и смысл числа там меняются вместе. */
function matchBlock(rate) {
  const box = el('div', 'match');
  box.append(
    el('b', null, Number.isFinite(Number(rate)) ? percent(rate) : '—'),
    el('span', null, 'команд из этой группы распознаватель отдаёт навыку'),
  );
  return box;
}

/** Обе кнопки блокируются на время запроса, так что второй клик не уходит. */
function decisionButtons(proposal, card) {
  const box = el('div', 'card-actions');
  const buttons = [];

  const decide = async (call) => {
    if (buttons.some((button) => button.disabled)) return;
    buttons.forEach((button) => { button.disabled = true; });
    try {
      await call(proposal.id);
      await dismiss(card);
      /* Принятый навык работает сразу — без перезапроса список навыков врёт. */
      await Promise.all([refreshSkills(), refreshProposals()]);
      notifyChange();
    } catch (error) {
      console.error(error);
      buttons.forEach((button) => { button.disabled = false; });
      toast('Не получилось. Попробуй ещё раз.');
    }
  };

  for (const [label, call, primary] of [
    ['Принять', api.acceptProposal, true],
    ['Отклонить', api.rejectProposal, false],
  ]) {
    const button = el('button', primary ? 'btn btn-primary' : 'btn', label);
    button.type = 'button';
    button.addEventListener('click', () => decide(call));
    buttons.push(button);
    box.append(button);
  }
  return box;
}

function dismiss(card) {
  if (reduced.matches) { card.remove(); return Promise.resolve(); }
  card.classList.add('is-leaving');
  return new Promise((resolve) => setTimeout(resolve, DISMISS_MS));
}

function proposalCard(proposal) {
  const skill = proposal.skill ?? {};
  const questions = skill.questions ?? {};
  const card = el('article', 'card proposal');

  const head = el('div', 'proposal-head');
  head.append(el('h3', null, `Новый навык: ${skill.name ?? skill.id ?? 'без имени'}`));
  head.append(matchBlock(proposal.match_rate));

  card.append(head, el('p', 'card-note', skill.description ?? ''));

  const examples = Array.isArray(skill.examples) ? skill.examples : [];
  if (examples.length) {
    card.append(el('h4', null, 'Из каких команд'), examplesList(examples));
  }

  const rules = Array.isArray(skill.rules) ? skill.rules : [];
  if (rules.length) {
    const list = el('ul', 'rules');
    list.append(...rules.map((rule) => ruleRow(rule, questions)));
    card.append(el('h4', null, 'Что будет делать'), list);
  }

  card.append(decisionButtons(proposal, card));
  return card;
}

function renderProposals(proposals) {
  if (!proposals.length) {
    notice(
      proposalsEl,
      'Pixel учится. Когда наберётся несколько похожих непонятых команд, здесь появится новый навык.',
    );
    return;
  }
  proposalsEl.replaceChildren(...proposals.map(proposalCard));
}

/* ─── Загрузка и обновление ─────────────────────────────────────────────── */

const changeListeners = new Set();

/** Метрики живут в app.js, но меняются от тех же событий, что и панель. */
export function onLearningChange(callback) {
  changeListeners.add(callback);
}

function notifyChange() {
  changeListeners.forEach((callback) => callback());
}

export async function refreshSkills() {
  try {
    const skills = await api.getSkills();
    renderSkills(Array.isArray(skills) ? skills : []);
  } catch (error) {
    console.error(error);
    notice(skillsEl, 'Не удалось загрузить навыки.');
  }
}

export async function refreshProposals() {
  try {
    const proposals = await api.getProposals();
    renderProposals(Array.isArray(proposals) ? proposals : []);
  } catch (error) {
    console.error(error);
    notice(proposalsEl, 'Не удалось загрузить предложения.');
  }
}

/* ─── Ручной запуск майнера ─────────────────────────────────────────────── */

async function runMiner() {
  mineEl.disabled = true;
  mineEl.classList.add('is-busy');
  try {
    const result = await api.mine();
    if (!result?.started) toast('Майнер уже работает');
    else if (Number(result.proposals) > 0) toast(`Нашлось предложений: ${result.proposals}`);
    else toast('Пока не из чего учиться');
    await refreshProposals();
    notifyChange();
  } catch (error) {
    console.error(error);
    toast('Не получилось запустить поиск навыков');
  } finally {
    mineEl.disabled = false;
    mineEl.classList.remove('is-busy');
  }
}

mineEl.addEventListener('click', runMiner);
