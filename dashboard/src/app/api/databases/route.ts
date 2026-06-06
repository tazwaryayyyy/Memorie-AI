import { NextRequest, NextResponse } from 'next/server';
import { access, readFile } from 'fs/promises';
import path from 'path';
import os from 'os';

const MEMOIRE_SERVER =
  process.env.MEMOIRE_SERVER_URL ?? 'http://localhost:6779';

function getAllowedPaths(): string[] {
  const raw = process.env.MEMOIRE_ALLOWED_PATHS ?? '';
  if (!raw.trim()) return [];
  return raw.split(',').map((p) => p.trim()).filter(Boolean);
}

function validateDbPath(dbPath: string): boolean {
  const allowed = getAllowedPaths();
  if (allowed.length === 0) return true; // no restriction = allow all (local dev default)
  const resolved = path.resolve(dbPath);
  return allowed.some((dir) => resolved.startsWith(path.resolve(dir)));
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

export async function GET() {
  const serverErr = await checkServerReachable();
  if (serverErr) {
    return NextResponse.json({ error: serverErr }, { status: 503 });
  }

  try {
    const dbPaths = await discoverDbPaths();
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
            fetch(`${MEMOIRE_SERVER}/export?${params}`),
            fetch(`${MEMOIRE_SERVER}/info?${params}`),
          ]);

          if (exportRes.ok) {
            const exportJson = await exportRes.json();
            memories = exportJson.memories ?? [];
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
  try {
    const body = await req.json();
    const { action, dbPath, query, text, id } = body;

    if (!dbPath) {
      return NextResponse.json({ error: 'dbPath is required' }, { status: 400 });
    }

    if (!validateDbPath(dbPath)) {
      return NextResponse.json({ error: 'dbPath not in allowed paths' }, { status: 403 });
    }

    const serverErr = await checkServerReachable();
    if (serverErr) {
      return NextResponse.json({ error: serverErr }, { status: 503 });
    }

    const ns = body.namespace ?? 'default';

    switch (action) {
      case 'remember': {
        if (!text) return NextResponse.json({ error: 'text is required' }, { status: 400 });
        const res = await fetch(`${MEMOIRE_SERVER}/remember`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ db: dbPath, ns, text }),
        });
        const data = await res.json();
        return NextResponse.json({ output: JSON.stringify(data) });
      }
      case 'recall': {
        if (!query) return NextResponse.json({ error: 'query is required' }, { status: 400 });
        const res = await fetch(`${MEMOIRE_SERVER}/recall`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ db: dbPath, ns, query, k: 10 }),
        });
        const data = await res.json();
        return NextResponse.json({ results: data.memories ?? [] });
      }
      case 'forget': {
        if (id === undefined) return NextResponse.json({ error: 'id is required' }, { status: 400 });
        const res = await fetch(`${MEMOIRE_SERVER}/forget`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ db: dbPath, ns, id }),
        });
        const data = await res.json();
        return NextResponse.json({ output: JSON.stringify(data) });
      }
      case 'clear': {
        const res = await fetch(`${MEMOIRE_SERVER}/clear`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ db: dbPath, ns }),
        });
        const data = await res.json();
        return NextResponse.json({ output: JSON.stringify(data) });
      }
      default:
        return NextResponse.json({ error: `Unsupported action: ${action}` }, { status: 400 });
    }
  } catch (error: unknown) {
    const msg = error instanceof Error ? error.message : String(error);
    return NextResponse.json({ error: msg }, { status: 500 });
  }
}
