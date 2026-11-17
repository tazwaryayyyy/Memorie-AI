// Request checks for the dashboard's API routes. This UI is EXPERIMENTAL and
// LOCAL-ONLY: it can read and delete memories, so its routes must not be
// reachable from other websites (CSRF) or via DNS rebinding.
//
// Pure functions only (no `next` imports) so they can be unit-tested with
// `node --test`; see tests/localGuard.test.mts.

export interface RequestFacts {
  method: string;
  host: string | null;
  origin: string | null;
  contentType: string | null;
  secFetchSite: string | null;
}

/** `localhost` or an IPv4/IPv6 loopback literal, with an optional port. */
export function isLoopbackHost(hostport: string): boolean {
  let host: string;
  if (hostport.startsWith('[')) {
    const end = hostport.indexOf(']');
    if (end < 0) return false;
    host = hostport.slice(1, end);
  } else {
    const idx = hostport.lastIndexOf(':');
    host = idx < 0 ? hostport : hostport.slice(0, idx);
    if (host.includes(':')) return false; // bare IPv6 is not a valid Host value
  }
  host = host.toLowerCase();
  if (host === 'localhost' || host === '::1') return true;
  const m = /^127\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$/.exec(host);
  return !!m && m.slice(1).every((o) => Number(o) <= 255);
}

export function isLoopbackOrigin(origin: string): boolean {
  const m = /^https?:\/\/([^/]+)/.exec(origin);
  return !!m && isLoopbackHost(m[1]);
}

/** Returns a human-readable reason to reject the request, or null if it is fine. */
export function localRequestProblem(f: RequestFacts): string | null {
  if (!f.host || !isLoopbackHost(f.host)) {
    return 'Host must be localhost (DNS-rebinding protection)';
  }
  if (f.origin !== null && !isLoopbackOrigin(f.origin)) {
    return 'cross-origin requests are not allowed';
  }
  if (f.secFetchSite === 'cross-site') {
    return 'cross-site requests are not allowed';
  }
  if (f.method !== 'GET' && f.method !== 'HEAD') {
    // Forcing application/json makes a cross-origin POST require a CORS
    // preflight, which this server never grants.
    const ct = (f.contentType ?? '').split(';')[0].trim().toLowerCase();
    if (ct !== 'application/json') {
      return 'Content-Type must be application/json';
    }
  }
  return null;
}
