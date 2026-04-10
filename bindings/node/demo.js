'use strict';
/**
 * Node.js demo — simulates an AI agent using Memoire.
 *
 *   npm install ffi-napi ref-napi
 *   node bindings/node/demo.js
 */

const Memoire = require('./index');

async function main() {
  const m = new Memoire(':memory:');

  console.log('=== Memoire Node.js Demo ===\n');

  const notes = [
    'Fixed a race condition in the payment idempotency key validation. Two concurrent requests with the same key could both pass the uniqueness check.',
    'Replaced synchronous bcrypt calls with async argon2id. Reduced auth endpoint p99 from 420ms to 38ms.',
    'Diagnosed N+1 query in the /users dashboard. Added a single JOIN query. Page load dropped from 3.8s to 290ms.',
    'Added Redis-based rate limiting to /api/reset-password at 5 requests per hour per IP address.',
    'Memory leak in the WebSocket connection handler — connections were not removed from the registry on client disconnect.',
  ];

  console.log('Storing memories...');
  for (const note of notes) {
    const n = m.remember(note);
    console.log(`  ✓ (${n} chunk) ${note.slice(0, 65)}…`);
  }
  console.log(`\nTotal chunks: ${m.count()}\n`);

  const queries = [
    'security vulnerabilities and auth fixes',
    'database query performance',
    'rate limiting',
    'memory leaks',
  ];

  for (const q of queries) {
    console.log(`Query: "${q}"`);
    const results = m.recall(q, 2);
    for (const r of results) {
      console.log(`  [${r.score.toFixed(4)}] ${r.content.slice(0, 85)}…`);
    }
    console.log();
  }

  m.close();
}

main().catch(console.error);
