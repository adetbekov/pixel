/* Мок-бэкенд под `?mock=1` — панель навыков разрабатывается и смотрится без FastAPI.

   Фикстуры лежат в `frontend/mocks/` и остаются в репозитории как документация
   формы ответов: `skills.json` и `proposals.json` повторяют `SkillCard` и
   `Proposal` из `backend/api.py` поле в поле.

   `?mock=1&busy=1` заставляет `POST /api/mine` ответить `started: false` —
   иначе эту ветку тоста нечем проверить. */

const params = new URLSearchParams(location.search);

export const MOCK = params.get('mock') === '1';

const MINER_BUSY = params.get('busy') === '1';

/* Задержка, чтобы спиннеры и блокировка кнопок были видны, а не мигали. */
const LATENCY_MS = 350;

const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

let store = null;

async function fixture(name) {
  const response = await fetch(`./mocks/${name}.json`);
  if (!response.ok) throw new Error(`mock ${name}.json -> ${response.status}`);
  return response.json();
}

async function load() {
  if (!store) {
    const [skills, proposals, metrics, state, history] = await Promise.all(
      ['skills', 'proposals', 'metrics', 'state', 'history'].map(fixture),
    );
    store = { skills, proposals, metrics, state, history };
  }
  return store;
}

function resolveProposal(id, accepted) {
  const index = store.proposals.findIndex((p) => p.id === id);
  if (index < 0) throw new Error(`mock: unknown proposal ${id}`);
  const [proposal] = store.proposals.splice(index, 1);
  if (!accepted) return;

  const { skill } = proposal;
  store.skills.push({
    id: skill.id,
    name: skill.name,
    description: skill.description,
    status: 'active',
    origin: 'mined',
    uses: 0,
    likes: 0,
    dislikes: 0,
  });
  store.metrics = { ...store.metrics, skills_active: store.metrics.skills_active + 1 };
}

const REPLY = {
  interaction_id: 1,
  reply: 'Это мок — Pixel отвечает заглушкой.',
  actions: [{ action: 'wave', args: {} }],
  engine: 'gemini',
  skill_id: null,
  confidence: 0.31,
  latency_ms: LATENCY_MS,
};

/** Разбирает `/proposals/{id}/accept` в `[id, 'accept']`; иначе `[]`. */
function proposalRoute(path) {
  const match = /^\/proposals\/([^/]+)\/(accept|reject)$/.exec(path);
  return match ? [decodeURIComponent(match[1]), match[2]] : [];
}

export async function mockRequest(method, path) {
  const data = await load();
  await wait(LATENCY_MS);

  if (method === 'GET') {
    if (path === '/state') return data.state;
    if (path === '/history') return data.history;
    if (path === '/skills') return data.skills;
    if (path === '/proposals') return data.proposals;
    if (path === '/metrics') return data.metrics;
  }

  if (method === 'POST') {
    const [id, verb] = proposalRoute(path);
    if (verb) {
      resolveProposal(id, verb === 'accept');
      return { ok: true };
    }
    if (path === '/mine') {
      if (MINER_BUSY) return { started: false, proposals: 0 };
      return { started: true, proposals: data.proposals.length };
    }
    if (path === '/feedback') return { ok: true };
    if (path === '/chat' || path === '/action') return { ...REPLY, state: data.state };
  }

  throw new Error(`mock: ${method} ${path} is not stubbed`);
}
