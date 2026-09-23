/* Робот: выражения лица, анимации примитивов, последовательная очередь планов. */

const FACES = ['happy', 'sad', 'sleepy', 'angry', 'curious'];

/* action -> имя @keyframes и длительность из styles.css */
const ANIMATIONS = {
  jump:  { name: 'jump',  ms: 600 },
  dance: { name: 'dance', ms: 1400 },
  sleep: { name: 'sleep', ms: 2000 },
  eat:   { name: 'eat',   ms: 900 },
  spin:  { name: 'spin',  ms: 800 },
  wave:  { name: 'wave',  ms: 800 },
};

const MAX_ACTIONS = 8;
const SPEECH_MS = 2500;

const stage = document.getElementById('robot-stage');
const svg = document.getElementById('pixel');
const speech = document.getElementById('speech');
const reduced = window.matchMedia('(prefers-reduced-motion: reduce)');

let chain = Promise.resolve();
let running = 0;
let speechTimer = 0;
const busyListeners = new Set();

function setBusy(delta) {
  running += delta;
  const busy = running > 0;
  busyListeners.forEach((cb) => cb(busy));
}

export function onBusyChange(cb) {
  busyListeners.add(cb);
}

export function setFace(face) {
  if (!FACES.includes(face)) return;
  FACES.forEach((f) => svg.classList.toggle(`face-${f}`, f === face));
}

export function say(text) {
  const value = text === undefined || text === null ? '…' : String(text).trim();
  if (!value) return;
  speech.textContent = value;
  speech.hidden = false;
  clearTimeout(speechTimer);
  speechTimer = setTimeout(() => { speech.hidden = true; }, SPEECH_MS);
}

function animate(action) {
  const anim = ANIMATIONS[action];
  const className = `anim-${action}`;

  return new Promise((resolve) => {
    if (reduced.matches) { resolve(); return; }

    let fallback = 0;
    const done = () => {
      stage.removeEventListener('animationend', onEnd);
      clearTimeout(fallback);
      stage.classList.remove(className);
      resolve();
    };
    const onEnd = (event) => { if (event.animationName === anim.name) done(); };

    stage.classList.remove(className);
    void stage.offsetWidth;                 // рестарт анимации при повторе действия
    stage.addEventListener('animationend', onEnd);
    stage.classList.add(className);
    fallback = setTimeout(done, anim.ms + 300);
  });
}

/* Принимает и {action, args}, и просто "jump". */
function normalize(item) {
  if (typeof item === 'string') return { action: item, args: {} };
  return { action: item?.action, args: item?.args ?? {} };
}

async function runPlan(actions) {
  for (const raw of actions.slice(0, MAX_ACTIONS)) {
    const { action, args } = normalize(raw);

    if (action === 'set_face') { setFace(args.face); continue; }
    if (action === 'say') { say(args.text); continue; }
    if (!ANIMATIONS[action]) continue;       // неизвестное действие просто пропускаем

    if (action === 'sleep') setFace('sleepy');
    await animate(action);
  }
}

/** Проигрывает план последовательно; параллельные вызовы встают в очередь. */
export function playPlan(actions) {
  const list = Array.isArray(actions) ? actions : [];
  setBusy(1);
  chain = chain
    .then(() => runPlan(list))
    .catch((err) => { console.error('playPlan', err); })
    .finally(() => setBusy(-1));
  return chain;
}
