import js from '@eslint/js';
import globals from 'globals';

// Статический гейт для frontend/: голые ES-модули, без сборки и без рантайм-зависимостей.
// Задача конфига — ловить то, что иначе всплывает только руками на QA: сломанный
// синтаксис, опечатки в именах (no-undef), мёртвые переменные, дубли ключей.
export default [
  {
    ignores: ['node_modules/**'],
  },
  {
    files: ['frontend/**/*.js'],
    languageOptions: {
      ecmaVersion: 2022,
      sourceType: 'module',
      globals: globals.browser,
    },
    rules: js.configs.recommended.rules,
  },
];
