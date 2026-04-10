'use strict';
/**
 * Minimal test suite for the Memoire Node.js binding.
 *
 * Usage:
 *   cargo build --release
 *   node bindings/node/test.js
 */

const Memoire = require('./index');

let passed = 0;
let failed = 0;

function assert(condition, msg) {
  if (condition) {
    console.log(`  ✓ ${msg}`);
    passed++;
  } else {
    console.error(`  ✗ ${msg}`);
    failed++;
  }
}

function assertThrows(fn, msg) {
  try {
    fn();
    console.error(`  ✗ ${msg} (expected throw, got none)`);
    failed++;
  } catch (_) {
    console.log(`  ✓ ${msg}`);
    passed++;
  }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

console.log('\n=== Memoire Node.js Tests ===\n');

{
  console.log('remember + count');
  const m = new Memoire(':memory:');
  assert(m.count() === 0, 'count starts at 0');
  const n = m.remember('Fixed the authentication bug in middleware');
  assert(n >= 1, `remember returns chunk count >= 1 (got ${n})`);
  assert(m.count() >= 1, 'count increments after remember');
  m.close();
}

{
  console.log('\nrecall');
  const m = new Memoire(':memory:');
  m.remember('Fixed null pointer dereference in auth middleware security patch');
  m.remember('Refactored database connection pool for throughput improvements');
  m.remember('Added unit tests for the payment processing module coverage');

  const results = m.recall('authentication security bug', 3);
  assert(Array.isArray(results), 'recall returns array');
  assert(results.length > 0, 'recall returns results');
  assert(typeof results[0].score === 'number', 'result has score');
  assert(typeof results[0].content === 'string', 'result has content');
  assert(typeof results[0].id === 'number', 'result has id');

  // Scores descending
  let descending = true;
  for (let i = 1; i < results.length; i++) {
    if (results[i].score > results[i - 1].score) descending = false;
  }
  assert(descending, 'scores are descending');
  m.close();
}

{
  console.log('\nempty store recall');
  const m = new Memoire(':memory:');
  const results = m.recall('anything', 5);
  assert(Array.isArray(results) && results.length === 0, 'empty store returns []');
  m.close();
}

{
  console.log('\nforget');
  const m = new Memoire(':memory:');
  m.remember('memory to delete');
  const results = m.recall('memory to delete', 1);
  const id = results[0].id;
  assert(m.forget(id) === true, 'forget returns true when deleted');
  assert(m.forget(id) === false, 'forget returns false when not found');
  m.close();
}

{
  console.log('\nclear');
  const m = new Memoire(':memory:');
  m.remember('one');
  m.remember('two');
  m.clear();
  assert(m.count() === 0, 'count is 0 after clear');
  m.close();
}

{
  console.log('\ntop_k limit');
  const m = new Memoire(':memory:');
  for (let i = 0; i < 8; i++) m.remember(`memory ${i} about coding in Rust`);
  const results = m.recall('Rust coding', 3);
  assert(results.length <= 3, `top_k=3 returns <= 3 results (got ${results.length})`);
  m.close();
}

// ─── Summary ──────────────────────────────────────────────────────────────────

console.log(`\n${'─'.repeat(40)}`);
console.log(`  ${passed} passed, ${failed} failed`);
if (failed > 0) process.exit(1);
