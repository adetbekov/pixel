/* Гейт контраста: поднимает frontend/index.html?mock=1 в headless Chromium и
   меряет контраст каждого текстового узла по вычисленным стилям.
 *
 * Почему браузер, а не разбор CSS: и фон, и цвет текста здесь собираются из
 * альфа-заливок поверх нескольких слоёв, а `opacity` предка умножается на всю
 * ветку. Статически это не считается — JEB-1521 как раз и прожил до QA потому,
 * что `opacity: .55` на карточке выглядела безобидно в диффе.
 *
 * Мок-режим backend'а не требует: `frontend/mocks/` содержит и отключённый
 * навык, и историю с уже проставленной оценкой, так что на экране оказываются
 * ровно те состояния, где контраст падал.
 *
 * Запуск: node .github/scripts/contrast_gate.mjs
 */

import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { extname, join, normalize } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

const ROOT = fileURLToPath(new URL('../../frontend/', import.meta.url));

const TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.svg': 'image/svg+xml',
};

/* Обе схемы: тёмная и светлая берут разные токены, и ниже AA падала светлая. */
const SCHEMES = ['light', 'dark'];

async function serve() {
  const server = createServer((req, res) => {
    const path = normalize(decodeURIComponent(new URL(req.url, 'http://x').pathname));
    const file = join(ROOT, path === '/' ? 'index.html' : path);
    if (!file.startsWith(ROOT)) {
      res.writeHead(403).end();
      return;
    }
    readFile(file).then(
      (body) => {
        res.writeHead(200, { 'content-type': TYPES[extname(file)] ?? 'application/octet-stream' });
        res.end(body);
      },
      () => res.writeHead(404).end(),
    );
  });
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
  return { server, port: server.address().port };
}

/* Всё, что ниже, выполняется в странице: без DOM эти числа не существуют. */
function collect() {
  const parse = (value) => {
    const [r, g, b, a = '1'] = value.replace(/^rgba?\(|\)$/g, '').split(/[\s,/]+/).filter(Boolean);
    return { r: +r, g: +g, b: +b, a: +a };
  };

  const over = (fg, bg) => ({
    r: fg.r * fg.a + bg.r * (1 - fg.a),
    g: fg.g * fg.a + bg.g * (1 - fg.a),
    b: fg.b * fg.a + bg.b * (1 - fg.a),
    a: 1,
  });

  const luminance = ({ r, g, b }) => {
    const channel = (v) => {
      const c = v / 255;
      return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
    };
    return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b);
  };

  const contrast = (x, y) => {
    const [hi, lo] = [luminance(x), luminance(y)].sort((a, b) => b - a);
    return (hi + 0.05) / (lo + 0.05);
  };

  const label = (el) => {
    const id = el.id ? `#${el.id}` : '';
    const cls = typeof el.className === 'string' && el.className ? `.${el.className.trim().split(/\s+/).join('.')}` : '';
    return `${el.tagName.toLowerCase()}${id}${cls}`;
  };

  const results = [];

  for (const el of document.body.querySelectorAll('*')) {
    /* Текст робота живёт в SVG и красится через fill — отдельная модель, и он
       декоративный: подписи под ним дублируют смысл. */
    if (el.closest('svg')) continue;

    const own = [...el.childNodes].some((n) => n.nodeType === Node.TEXT_NODE && n.textContent.trim());
    if (!own) continue;
    if (el.closest('.visually-hidden')) continue;

    const style = getComputedStyle(el);
    if (style.visibility === 'hidden' || style.display === 'none') continue;
    if (!el.getClientRects().length) continue;

    /* Слои от корня вниз: у каждого своя альфа фона, помноженная на
       накопленную opacity — ровно так их складывает и композитор браузера. */
    const chain = [];
    for (let node = el; node && node !== document.documentElement.parentElement; node = node.parentElement) {
      chain.unshift(node);
    }

    let accumulated = 1;
    let background = { r: 255, g: 255, b: 255, a: 1 };
    const opacities = [];
    for (const node of chain) {
      accumulated *= Number(getComputedStyle(node).opacity);
      opacities.push(accumulated);
      const layer = parse(getComputedStyle(node).backgroundColor);
      if (layer.a > 0) background = over({ ...layer, a: layer.a * accumulated }, background);
    }

    const effective = opacities[opacities.length - 1];
    const color = parse(style.color);
    if (color.a * effective === 0) continue;

    const ratio = contrast(over({ ...color, a: color.a * effective }, background), background);

    const size = parseFloat(style.fontSize);
    const weight = Number(style.fontWeight) || 400;
    const large = size >= 24 || (size >= 18.66 && weight >= 700);

    results.push({
      selector: label(el),
      text: el.textContent.trim().slice(0, 60),
      size,
      weight,
      ratio: Math.round(ratio * 100) / 100,
      threshold: large ? 3 : 4.5,
    });
  }

  return results;
}

const { server, port } = await serve();
const browser = await chromium.launch();
const failures = [];
let checked = 0;

try {
  for (const colorScheme of SCHEMES) {
    const context = await browser.newContext({ colorScheme, viewport: { width: 1440, height: 900 } });
    const page = await context.newPage();

    const errors = [];
    page.on('pageerror', (error) => errors.push(String(error)));

    await page.goto(`http://127.0.0.1:${port}/index.html?mock=1`);
    /* Карточки приезжают из мока с задержкой; без этого гейт измерял бы пустую
       страницу и был бы зелёным всегда. */
    await page.waitForSelector('.skill.is-off');
    await page.waitForSelector('.vote button[disabled]');
    await page.waitForSelector('.proposal');

    if (errors.length) throw new Error(`ошибка страницы (${colorScheme}): ${errors[0]}`);

    const nodes = await page.evaluate(collect);
    if (nodes.length < 20) throw new Error(`${colorScheme}: измерено ${nodes.length} узлов — страница не отрисовалась`);

    checked += nodes.length;
    for (const node of nodes) {
      if (node.ratio < node.threshold) failures.push({ colorScheme, ...node });
    }

    await context.close();
  }
} finally {
  await browser.close();
  server.close();
}

if (failures.length) {
  console.error(`Контраст ниже WCAG AA — ${failures.length} узл(ов):\n`);
  for (const f of failures) {
    console.error(
      `  [${f.colorScheme}] ${f.selector} — ${f.ratio}:1 при пороге ${f.threshold} ` +
        `(${f.size}px/${f.weight}) «${f.text}»`,
    );
  }
  process.exit(1);
}

console.log(`Контраст AA: ${checked} текстовых узлов в схемах ${SCHEMES.join(', ')} — порог держат.`);
