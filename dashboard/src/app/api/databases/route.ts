import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import os from 'os';
import { execFile } from 'child_process';
import { promisify } from 'util';

const execFileAsync = promisify(execFile);

// Helper to locate the compiled Rust CLI executable
function getCliPath(): string {
  const root = path.resolve(process.cwd(), '..');
  const isWindows = os.platform() === 'win32';
  const binName = isWindows ? 'memoire.exe' : 'memoire';

  const releasePath = path.join(root, 'target', 'release', binName);
  const debugPath = path.join(root, 'target', 'debug', binName);

  if (fs.existsSync(releasePath)) {
    return releasePath;
  }
  if (fs.existsSync(debugPath)) {
    return debugPath;
  }
  // Try system PATH fallback
  return binName;
}

// Helper to discover databases from telemetry logs
function discoverDbPaths(): string[] {
  const logPath = path.join(os.homedir(), '.memoire', 'logs', 'mcp-server.jsonl');
  const defaultPath = path.resolve(process.cwd(), '..', 'memoire.db');
  const pathsSet = new Set<string>([defaultPath]);

  if (fs.existsSync(logPath)) {
    try {
      const content = fs.readFileSync(logPath, 'utf8');
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

  return Array.from(pathsSet).filter(p => fs.existsSync(p) || p.endsWith('memoire.db'));
}

export async function GET() {
  try {
    const cli = getCliPath();
    const dbPaths = discoverDbPaths();
    const databases = [];

    for (const dbPath of dbPaths) {
      let memories = [];
      let infoText = '';
      let error = null;

      try {
        if (fs.existsSync(dbPath)) {
          // Export memories to JSON using Rust CLI
          const { stdout: exportOut } = await execFileAsync(cli, ['--db', dbPath, 'export', '--json']);
          memories = JSON.parse(exportOut.trim() || '[]');

          // Get stats using info command
          const { stdout: infoOut } = await execFileAsync(cli, ['--db', dbPath, 'info']);
          infoText = infoOut.trim();
        }
      } catch (err: any) {
        error = err.message;
      }

      databases.push({
        path: dbPath,
        name: path.basename(dbPath),
        exists: fs.existsSync(dbPath),
        memories,
        info: infoText,
        error,
      });
    }

    return NextResponse.json({ databases });
  } catch (error: any) {
    return NextResponse.json({ error: error.message }, { status: 500 });
  }
}

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const { action, dbPath, query, text, id, confirm } = body;

    if (!dbPath) {
      return NextResponse.json({ error: 'dbPath is required' }, { status: 400 });
    }

    const cli = getCliPath();
    let args: string[] = ['--db', dbPath];

    switch (action) {
      case 'remember':
        if (!text) return NextResponse.json({ error: 'text is required' }, { status: 400 });
        args.push('remember', text);
        break;
      case 'recall':
        if (!query) return NextResponse.json({ error: 'query is required' }, { status: 400 });
        args.push('recall', query, '--json');
        break;
      case 'forget':
        if (id === undefined) return NextResponse.json({ error: 'id is required' }, { status: 400 });
        args.push('forget', String(id));
        break;
      case 'clear':
        args.push('clear');
        if (confirm) {
          args.push('--confirm');
        }
        break;
      default:
        return NextResponse.json({ error: `Unsupported action: ${action}` }, { status: 400 });
    }

    const { stdout, stderr } = await execFileAsync(cli, args);

    if (action === 'recall') {
      return NextResponse.json({ results: JSON.parse(stdout.trim() || '[]'), stderr });
    }

    return NextResponse.json({ output: stdout.trim(), stderr });
  } catch (error: any) {
    return NextResponse.json({ error: error.message }, { status: 500 });
  }
}
