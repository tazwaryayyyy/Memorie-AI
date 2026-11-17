import { NextRequest, NextResponse } from 'next/server';
import { access, readFile } from 'fs/promises';
import path from 'path';
import os from 'os';
import { guardResponse } from '@/lib/guardResponse';

// EXPERIMENTAL, LOCAL-ONLY. See the guard in src/lib/localGuard.ts.

const MEMOIRE_SERVER =
  process.env.MEMOIRE_SERVER_URL ?? 'http://localhost:6779';

/**
 * Directories the dashboard may touch: MEMOIRE_ALLOWED_PATHS (comma-separated)
 * or, when unset, ONLY the repository root (the parent of dashboard/). There is
 * no "allow everything" default. memoire-server enforces its own sandbox on the
 * canonical path (symlinks resolved); this check is a first line of defence.
 */
function getAllowedRoots(): string[] {
  const roots = (process.env.MEMOIRE_ALLOWED_PATHS ?? '')
    .split(',')
    .map((p) => p.trim())
    .filter(Boolean)
    .map((p) => path.resolve(p));
  return roots.length > 0 ? roots : [path.resolve(process.cwd(), '..')];
}

function isInside(root: string, target: string): boolean {
  const rel = path.relative(root, target);
  return (
    rel !== '' &&
    rel !== '..' &&
    !rel.startsWith('..' + path.sep) &&
    !path.isAbsolute(rel)
  );
}

function validateDbPath(dbPath: string): boolean {
  const resolved = path.resolve(dbPath);
  return getAllowedRoots().some((root) => isInside(root, resolved));
}

function serverHeaders(json: boolean): Record<string, string> {
  const headers: Record<string, string> = {};
  if (json) headers['Content-Type'] = 'application/json';
  const token = process.env.MEMOIRE_API_TOKEN?.trim();
  if (token) headers['Authorization'] = `Bearer ${token}`;
  return headers;
}

async function callServer(
  pathname: string,
  body: unknown,
): Promise<{ ok: boolean; status: number; data: Record<string, unknown> }> {
  const res = await fetch(`${MEMOIRE_SERVER}${pathname}`, {
    method: 'POST',
    headers: serverHeaders(true),
    body: JSON.stringify(body),
  });
  const data = (await res.json().catch(() => ({}))) as Record<string, unknown>;
  return { ok: res.ok, status: res.status, data };
}

function serverError(r: { status: number; data: Record<string, unknown> }) {
  const msg = typeof r.data.error === 'string' ? r.data.error : `memoire-server returned ${r.status}`;
  return NextResponse.json({ error: msg }, { status: r.status });
}

// Check that memoire-server is reachable. Returns null on success, error message on failure.
async function checkServerReachable(): Promise<string | null> {
  try {
    const res = await fetch(`${MEMOIRE_SERVER}/health`, { signal: AbortSignal.timeout(2000) });
    if (!res.ok) return `Server returned ${res.status}`;
    return null;
  } catch {
    return 'memoire-server is unreachable. Start it first: ./target/release/memoire-server';
  }
}

// Helper to discover databases from telemetry logs
async function discoverDbPaths(): Promise<string[]> {
  const logPath = path.join(os.homedir(), '.memoire', 'logs', 'mcp-server.jsonl');
  const defaultPath = path.resolve(process.cwd(), '..', 'memoire.db');
  const pathsSet = new Set<string>([defaultPath]);

  const logExists = await access(logPath).then(() => true).catch(() => false);
  if (logExists) {
    try {
      const content = await readFile(logPath, 'utf8');
      const lines = content.split('\n');
      for (const line of lines) {
        if (line.trim()) {
          try {
            const parsed = JSON.parse(line);
            if (parsed.db_path) {
              pathsSet.add(path.resolve(parsed.db_path));
            }
          } catch {
            // Ignore invalid lines
          }
        }
      }
    } catch {
      // Ignore reading issues
    }
  }

  const results: string[] = [];
  for (const p of pathsSet) {
    const exists = await access(p).then(() => true).catch(() => false);
    if (exists || p.endsWith('memoire.db')) {
      results.push(p);
    }
  }
  return results;
}

export async function GET(req: NextRequest) {
  const blocked = guardResponse(req);
  if (blocked) return blocked;

  const serverErr = await checkServerReachable();
  if (serverErr) {
    return NextResponse.json({ error: serverErr }, { status: 503 });
  }

  try {
    // Paths come from MCP telemetry logs, so treat them as untrusted input.
    const dbPaths = (await discoverDbPaths()).filter(validateDbPath);
    const databases = [];

    for (const dbPath of dbPaths) {
      let memories: unknown[] = [];
      let infoData: Record<string, unknown> = {};
      let error: string | null = null;

      try {
        const dbExists = await access(dbPath).then(() => true).catch(() => false);
        if (dbExists) {
          const params = new URLSearchParams({ db: dbPath, ns: 'default' });

          const [exportRes, infoRes] = await Promise.all([
            fetch(`${MEMOIRE_SERVER}/export?${params}`, { headers: serverHeaders(false) }),
            fetch(`${MEMOIRE_SERVER}/info?${params}`, { headers: serverHeaders(false) }),
          ]);

          if (exportRes.ok) {
            const exportJson = await exportRes.json();
            memories = exportJson.memories ?? [];
          } else {
            const body = (await exportRes.json().catch(() => ({}))) as { error?: string };
            error = body.error ?? `memoire-server returned ${exportRes.status}`;
          }
          if (infoRes.ok) {
            infoData = await infoRes.json();
          }
        }
      } catch (err: unknown) {
        error = err instanceof Error ? err.message : String(err);
      }

      const exists = await access(dbPath).then(() => true).catch(() => false);
      databases.push({
        path: dbPath,
        name: path.basename(dbPath),
        exists,
        memories,
        info: infoData,
        error,
      });
    }

    return NextResponse.json({ databases });
  } catch (error: unknown) {
    const msg = error instanceof Error ? error.message : String(error);
    return NextResponse.json({ error: msg }, { status: 500 });
  }
}

export async function POST(req: NextRequest) {
  const blocked = guardResponse(req);
  if (blocked) return blocked;

  try {
    const body = await req.json();
    const { action, dbPath, query, text, id } = body;

    if (typeof dbPath !== 'string' || !dbPath) {
      return NextResponse.json({ error: 'dbPath is required' }, { status: 400 });
    }

    if (!validateDbPath(dbPath)) {
      return NextResponse.json({ error: 'dbPath not in allowed paths' }, { status: 403 });
    }

    const serverErr = await checkServerReachable();
    if (serverErr) {
      return NextResponse.json({ error: serverErr }, { status: 503 });
    }

    const ns = typeof body.namespace === 'string' && body.namespace ? body.namespace : 'default';

    // NOTE: there is deliberately no 'clear' action. The UI never used it, and
    // wiping a namespace is an admin operation (see SECURITY.md).
    switch (action) {
      case 'remember': {
        if (!text) return NextResponse.json({ error: 'text is required' }, { status: 400 });
        const r = await callServer('/remember', { db: dbPath, ns, text });
        if (!r.ok) return serverError(r);
        return NextResponse.json({ output: JSON.stringify(r.data) });
      }
      case 'recall': {
        if (!query) return NextResponse.json({ error: 'query is required' }, { status: 400 });
        const r = await callServer('/recall', { db: dbPath, ns, query, k: 10 });
        if (!r.ok) return serverError(r);
        return NextResponse.json({ results: r.data.memories ?? [] });
      }
      case 'forget': {
        if (id === undefined) return NextResponse.json({ error: 'id is required' }, { status: 400 });
        const r = await callServer('/forget', { db: dbPath, ns, id });
        if (!r.ok) return serverError(r);
        return NextResponse.json({ output: JSON.stringify(r.data) });
      }
      default:
        return NextResponse.json({ error: `Unsupported action: ${action}` }, { status: 400 });
    }
  } catch (error: unknown) {
    const msg = error instanceof Error ? error.message : String(error);
    return NextResponse.json({ error: msg }, { status: 500 });
  }
}
