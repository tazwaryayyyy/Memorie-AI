import { NextResponse } from 'next/server';
import { localRequestProblem } from './localGuard';

/** Returns a 403 response if the request fails the local-only checks, else null. */
export function guardResponse(req: Request): NextResponse | null {
  const problem = localRequestProblem({
    method: req.method,
    host: req.headers.get('host'),
    origin: req.headers.get('origin'),
    contentType: req.headers.get('content-type'),
    secFetchSite: req.headers.get('sec-fetch-site'),
  });
  return problem ? NextResponse.json({ error: problem }, { status: 403 }) : null;
}
