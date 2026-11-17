import test from 'node:test';
import assert from 'node:assert/strict';
import {
  isLoopbackHost,
  isLoopbackOrigin,
  localRequestProblem,
  type RequestFacts,
} from '../src/lib/localGuard.ts';

const base: RequestFacts = {
  method: 'GET',
  host: 'localhost:3000',
  origin: null,
  contentType: null,
  secFetchSite: null,
};

test('loopback hosts are accepted', () => {
  for (const h of ['localhost', 'localhost:3000', 'LOCALHOST:1', '127.0.0.1:3000', '127.1.2.3', '[::1]', '[::1]:3000']) {
    assert.equal(isLoopbackHost(h), true, h);
  }
});

test('non-loopback and malformed hosts are rejected', () => {
  for (const h of ['', 'evil.example', 'evil.example:3000', 'localhost.evil.example', '127.0.0.1.evil.example', '127.0.0.999', '0.0.0.0:3000', '192.168.1.5', '::1', '[::1']) {
    assert.equal(isLoopbackHost(h), false, JSON.stringify(h));
  }
});

test('origins', () => {
  assert.equal(isLoopbackOrigin('http://localhost:3000'), true);
  assert.equal(isLoopbackOrigin('https://evil.example'), false);
  assert.equal(isLoopbackOrigin('null'), false);
  assert.equal(isLoopbackOrigin('http://localhost.evil.example'), false);
});

test('GET from the local page passes', () => {
  assert.equal(localRequestProblem(base), null);
  assert.equal(localRequestProblem({ ...base, origin: 'http://localhost:3000', secFetchSite: 'same-origin' }), null);
});

test('DNS rebinding (foreign Host) is rejected, as is a missing Host', () => {
  assert.match(localRequestProblem({ ...base, host: 'attacker.example:3000' }) ?? '', /Host/);
  assert.match(localRequestProblem({ ...base, host: null }) ?? '', /Host/);
});

test('cross-origin and cross-site are rejected', () => {
  assert.match(localRequestProblem({ ...base, origin: 'https://evil.example' }) ?? '', /cross-origin/);
  assert.match(localRequestProblem({ ...base, secFetchSite: 'cross-site' }) ?? '', /cross-site/);
});

test('POST must be application/json (blocks no-cors text/plain CSRF)', () => {
  const post = { ...base, method: 'POST' };
  assert.match(localRequestProblem({ ...post, contentType: 'text/plain' }) ?? '', /Content-Type/);
  assert.match(localRequestProblem({ ...post, contentType: null }) ?? '', /Content-Type/);
  assert.match(localRequestProblem({ ...post, contentType: 'application/x-www-form-urlencoded' }) ?? '', /Content-Type/);
  assert.equal(localRequestProblem({ ...post, contentType: 'application/json' }), null);
  assert.equal(localRequestProblem({ ...post, contentType: 'Application/JSON; charset=utf-8' }), null);
});
