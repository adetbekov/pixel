/* Тонкая обёртка над HTTP-контрактом бэкенда (см. JEB-1497). */

import { MOCK, mockRequest } from './mock.js';

const BASE = '/api';

async function request(path, options = {}) {
  const method = options.method ?? 'GET';
  if (MOCK) return mockRequest(method, path);

  const response = await fetch(`${BASE}${path}`, {
    headers: options.body ? { 'Content-Type': 'application/json' } : undefined,
    ...options,
  });
  if (!response.ok) {
    throw new Error(`${method} ${path} -> ${response.status}`);
  }
  return response.json();
}

const post = (path, body) =>
  request(path, { method: 'POST', body: body === undefined ? undefined : JSON.stringify(body) });

export const getState     = () => request('/state');
export const chat         = (text) => post('/chat', { text });
export const action       = (name) => post('/action', { name });
export const feedback     = (interactionId, value) => post('/feedback', { interaction_id: interactionId, value });
export const getHistory   = () => request('/history');
export const getSkills    = () => request('/skills');
export const getProposals = () => request('/proposals');
export const getMetrics   = () => request('/metrics');
export const mine         = () => post('/mine');
export const acceptProposal = (id) => post(`/proposals/${encodeURIComponent(id)}/accept`);
export const rejectProposal = (id) => post(`/proposals/${encodeURIComponent(id)}/reject`);
