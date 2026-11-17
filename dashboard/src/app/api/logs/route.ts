import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import os from 'os';
import { guardResponse } from '@/lib/guardResponse';

export async function GET(req: Request) {
  const blocked = guardResponse(req);
  if (blocked) return blocked;

  try {
    const logPath = path.join(os.homedir(), '.memoire', 'logs', 'mcp-server.jsonl');

    if (!fs.existsSync(logPath)) {
      return NextResponse.json({ logs: [] });
    }

    const content = fs.readFileSync(logPath, 'utf8');
    const lines = content.split('\n');
    const logs = [];

    // Parse lines from last to first
    for (let i = lines.length - 1; i >= 0; i--) {
      const line = lines[i].trim();
      if (line) {
        try {
          logs.push(JSON.parse(line));
        } catch {
          // Ignore invalid JSON lines
        }
      }
      if (logs.length >= 500) {
        break;
      }
    }

    return NextResponse.json({ logs });
  } catch (error: any) {
    return NextResponse.json({ error: error.message }, { status: 500 });
  }
}
