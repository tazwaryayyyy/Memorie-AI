'use client';

import React, { useState, useEffect } from 'react';
import { 
  Database, 
  Terminal, 
  Search, 
  Plus, 
  Trash2, 
  RefreshCw, 
  FileText, 
  AlertTriangle, 
  CheckCircle,
  HelpCircle,
  Shield,
  Layers,
  History,
  TrendingDown,
  Info
} from 'lucide-react';

interface Memory {
  id: number;
  content: string;
  score: number;
  trust: number;
  uncertainty: number;
  state: string;
  created_at: number;
  contradiction_group?: number;
}

interface DatabaseInfo {
  path: string;
  name: string;
  exists: boolean;
  memories: Memory[];
  info: string;
  error: string | null;
}

interface LogEntry {
  timestamp: string;
  levelname: string;
  name: string;
  message: string;
  event?: string;
  db_path?: string;
  query?: string;
  top_k?: number;
  results?: any[];
  memory_id?: number;
  reinforced?: boolean;
  outcomes?: any[];
  [key: string]: any;
}

export default function Dashboard() {
  const [databases, setDatabases] = useState<DatabaseInfo[]>([]);
  const [selectedDbPath, setSelectedDbPath] = useState<string>('');
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [loadingDbs, setLoadingDbs] = useState<boolean>(true);
  const [loadingLogs, setLoadingLogs] = useState<boolean>(true);
  const [autoRefresh, setAutoRefresh] = useState<boolean>(true);

  // Recall Test state
  const [recallQuery, setRecallQuery] = useState<string>('');
  const [recallResults, setRecallResults] = useState<Memory[]>([]);
  const [testingRecall, setTestingRecall] = useState<boolean>(false);

  // New Memory state
  const [newMemoryText, setNewMemoryText] = useState<string>('');
  const [addingMemory, setAddingMemory] = useState<boolean>(false);

  const selectedDb = databases.find(db => db.path === selectedDbPath) || databases[0];

  const fetchDatabases = async () => {
    try {
      const res = await fetch('/api/databases');
      const data = await res.json();
      if (data.databases) {
        setDatabases(data.databases);
        if (!selectedDbPath && data.databases.length > 0) {
          setSelectedDbPath(data.databases[0].path);
        }
      }
    } catch (err) {
      console.error('Failed to fetch databases', err);
    } finally {
      setLoadingDbs(false);
    }
  };

  const fetchLogs = async () => {
    try {
      const res = await fetch('/api/logs');
      const data = await res.json();
      if (data.logs) {
        setLogs(data.logs);
      }
    } catch (err) {
      console.error('Failed to fetch logs', err);
    } finally {
      setLoadingLogs(false);
    }
  };

  const loadData = () => {
    fetchDatabases();
    fetchLogs();
  };

  useEffect(() => {
    loadData();
  }, []);

  useEffect(() => {
    if (!autoRefresh) return;
    const interval = setInterval(loadData, 5000);
    return () => clearInterval(interval);
  }, [autoRefresh, selectedDbPath]);

  const handleRemember = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newMemoryText.trim() || !selectedDbPath) return;
    setAddingMemory(true);
    try {
      const res = await fetch('/api/databases', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          action: 'remember',
          dbPath: selectedDbPath,
          text: newMemoryText,
        }),
      });
      const data = await res.json();
      if (data.error) {
        alert(data.error);
      } else {
        setNewMemoryText('');
        fetchDatabases();
      }
    } catch (err: any) {
      alert(err.message);
    } finally {
      setAddingMemory(false);
    }
  };

  const handleRecallTest = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!recallQuery.trim() || !selectedDbPath) return;
    setTestingRecall(true);
    try {
      const res = await fetch('/api/databases', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          action: 'recall',
          dbPath: selectedDbPath,
          query: recallQuery,
        }),
      });
      const data = await res.json();
      if (data.error) {
        alert(data.error);
      } else {
        setRecallResults(data.results || []);
      }
    } catch (err: any) {
      alert(err.message);
    } finally {
      setTestingRecall(false);
    }
  };

  const handleDeleteMemory = async (id: number) => {
    if (!selectedDbPath || !confirm(`Are you sure you want to forget memory id=${id}?`)) return;
    try {
      const res = await fetch('/api/databases', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          action: 'forget',
          dbPath: selectedDbPath,
          id,
        }),
      });
      const data = await res.json();
      if (data.error) {
        alert(data.error);
      } else {
        fetchDatabases();
        // Clear matching recall item if present
        setRecallResults(prev => prev.filter(item => item.id !== id));
      }
    } catch (err: any) {
      alert(err.message);
    }
  };

  const getLogColor = (entry: LogEntry) => {
    if (entry.levelname === 'ERROR') return 'border-red-500 bg-red-950/20 text-red-400';
    if (entry.levelname === 'WARNING') return 'border-amber-500 bg-amber-950/20 text-amber-400';
    if (entry.event === 'memoire_reinforce') return 'border-emerald-500 bg-emerald-950/20 text-emerald-400';
    if (entry.event === 'memoire_penalize') return 'border-purple-500 bg-purple-950/20 text-purple-400';
    return 'border-zinc-800 bg-zinc-900/50 text-zinc-300';
  };

  const getLogBadge = (entry: LogEntry) => {
    if (entry.event) {
      return entry.event.replace('memoire_', '').toUpperCase();
    }
    return entry.levelname;
  };

  const getTrustBgColor = (trust: number) => {
    if (trust >= 0.7) return 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20';
    if (trust >= 0.4) return 'bg-amber-500/10 text-amber-400 border border-amber-500/20';
    return 'bg-red-500/10 text-red-400 border border-red-500/20';
  };

  const getStateBadgeColor = (state: string) => {
    switch (state) {
      case 'active':
        return 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20';
      case 'shadow':
        return 'bg-blue-500/10 text-blue-400 border border-blue-500/20';
      case 'archived':
        return 'bg-zinc-500/10 text-zinc-400 border border-zinc-500/20';
      case 'rejected':
        return 'bg-purple-500/10 text-purple-400 border border-purple-500/20';
      default:
        return 'bg-zinc-800 text-zinc-300';
    }
  };

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100 flex flex-col font-sans select-none antialiased">
      {/* Header */}
      <header className="border-b border-zinc-800/80 bg-zinc-900/40 backdrop-blur-md sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg bg-gradient-to-tr from-violet-600 to-indigo-600 flex items-center justify-center shadow-lg shadow-violet-500/10">
              <Layers className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="font-bold text-lg leading-tight tracking-tight bg-gradient-to-r from-zinc-50 to-zinc-400 bg-clip-text text-transparent">
                Memoire AI
              </h1>
              <p className="text-[10px] text-zinc-500 font-semibold tracking-widest uppercase">
                Observability Dashboard
              </p>
            </div>
          </div>

          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2 bg-zinc-900/80 border border-zinc-800 px-3 py-1.5 rounded-full text-xs">
              <span className={`w-2 h-2 rounded-full ${autoRefresh ? 'bg-emerald-500 animate-pulse' : 'bg-zinc-600'}`} />
              <span className="text-zinc-400 font-medium">Auto-Refresh</span>
              <button 
                onClick={() => setAutoRefresh(!autoRefresh)}
                className="ml-1 text-zinc-500 hover:text-zinc-300 transition-colors cursor-pointer"
              >
                {autoRefresh ? 'Disable' : 'Enable'}
              </button>
            </div>

            <button 
              onClick={loadData}
              className="p-2 bg-zinc-900 hover:bg-zinc-800 border border-zinc-800 hover:border-zinc-700 rounded-lg text-zinc-400 hover:text-zinc-100 transition-all cursor-pointer flex items-center justify-center"
              title="Refresh telemetry"
            >
              <RefreshCw className="w-4 h-4" />
            </button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-6 py-8 grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        {/* Left Column: Databases & Stats */}
        <section className="lg:col-span-4 flex flex-col gap-6">
          {/* Active Databases Selection */}
          <div className="bg-zinc-900/30 backdrop-blur-md border border-zinc-800/80 rounded-2xl p-5 flex flex-col gap-4">
            <div className="flex items-center gap-2">
              <Database className="w-4 h-4 text-violet-400" />
              <h2 className="font-semibold text-sm text-zinc-300">Memory Databases</h2>
            </div>
            
            {loadingDbs ? (
              <div className="text-zinc-500 text-xs py-2 flex items-center gap-2">
                <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                Scanning storage...
              </div>
            ) : databases.length === 0 ? (
              <p className="text-zinc-500 text-xs py-2">No active databases detected yet.</p>
            ) : (
              <div className="flex flex-col gap-1.5">
                {databases.map(db => (
                  <button
                    key={db.path}
                    onClick={() => setSelectedDbPath(db.path)}
                    className={`text-left px-3.5 py-3 rounded-xl border text-xs font-medium flex flex-col gap-1 transition-all cursor-pointer ${
                      selectedDbPath === db.path 
                        ? 'border-violet-500/40 bg-violet-600/5 text-violet-300 shadow-md shadow-violet-500/5' 
                        : 'border-zinc-800/80 bg-zinc-900/10 text-zinc-400 hover:border-zinc-700 hover:bg-zinc-900/30'
                    }`}
                  >
                    <div className="flex items-center justify-between w-full">
                      <span className="font-semibold truncate max-w-[180px]">{db.name}</span>
                      <span className="text-[10px] bg-zinc-800/80 px-2 py-0.5 rounded-md font-bold uppercase text-zinc-400">
                        {db.memories.length} Chunks
                      </span>
                    </div>
                    <span className="text-[10px] text-zinc-600 truncate w-full font-mono">{db.path}</span>
                  </button>
                ))}
              </div>
            )}
          </div>

          {/* Database Specs Card */}
          {selectedDb && (
            <div className="bg-zinc-900/30 backdrop-blur-md border border-zinc-800/80 rounded-2xl p-5 flex flex-col gap-4">
              <div className="flex items-center gap-2">
                <Info className="w-4 h-4 text-zinc-400" />
                <h3 className="font-semibold text-sm text-zinc-300">Storage Information</h3>
              </div>
              <div className="font-mono text-[11px] text-zinc-400 flex flex-col gap-2.5 bg-zinc-950/40 p-4 rounded-xl border border-zinc-900">
                <div className="flex justify-between border-b border-zinc-900 pb-1.5">
                  <span className="text-zinc-600">ENGINE</span>
                  <span className="text-zinc-300 font-medium">Rust SQLite Core</span>
                </div>
                <div className="flex justify-between border-b border-zinc-900 pb-1.5">
                  <span className="text-zinc-600">EMBED MODEL</span>
                  <span className="text-zinc-300 font-medium">all-MiniLM-L6-v2</span>
                </div>
                <div className="flex justify-between border-b border-zinc-900 pb-1.5">
                  <span className="text-zinc-600">DIMENSIONS</span>
                  <span className="text-zinc-300 font-medium">384-dim</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-zinc-600">FILE PATH</span>
                  <span className="text-zinc-400 truncate max-w-[200px]" title={selectedDb.path}>{selectedDb.name}</span>
                </div>
              </div>
            </div>
          )}

          {/* Write Panel */}
          {selectedDb && (
            <div className="bg-zinc-900/30 backdrop-blur-md border border-zinc-800/80 rounded-2xl p-5 flex flex-col gap-4">
              <div className="flex items-center gap-2">
                <Plus className="w-4 h-4 text-emerald-400" />
                <h3 className="font-semibold text-sm text-zinc-300">Store New Memory</h3>
              </div>
              <form onSubmit={handleRemember} className="flex flex-col gap-2.5">
                <textarea
                  value={newMemoryText}
                  onChange={(e) => setNewMemoryText(e.target.value)}
                  placeholder="Insert lessons, bug summaries, or decisions..."
                  className="w-full h-24 bg-zinc-950/80 border border-zinc-800 rounded-xl px-3 py-2 text-xs text-zinc-300 placeholder-zinc-600 focus:outline-none focus:border-violet-500/50 resize-none font-medium"
                />
                <button
                  type="submit"
                  disabled={addingMemory || !newMemoryText.trim()}
                  className="w-full py-2 bg-gradient-to-r from-violet-600 to-indigo-600 hover:from-violet-500 hover:to-indigo-500 disabled:opacity-40 disabled:pointer-events-none rounded-xl text-xs font-semibold text-white transition-all shadow-md shadow-violet-500/10 cursor-pointer flex items-center justify-center gap-1.5"
                >
                  {addingMemory ? 'Storing chunk...' : 'Store Memory'}
                </button>
              </form>
            </div>
          )}
        </section>

        {/* Right Column: Database Memory Logs, Query tests & Explore */}
        <section className="lg:col-span-8 flex flex-col gap-8">
          {/* Databases recall tester & explorer */}
          {selectedDb && (
            <div className="bg-zinc-900/30 backdrop-blur-md border border-zinc-800/80 rounded-2xl p-6 flex flex-col gap-6">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2.5">
                  <Database className="w-4.5 h-4.5 text-violet-400" />
                  <div>
                    <h3 className="font-bold text-base text-zinc-100">{selectedDb.name}</h3>
                    <p className="text-[10px] text-zinc-500 font-mono mt-0.5">{selectedDb.path}</p>
                  </div>
                </div>
              </div>

              {/* Recall testing panel */}
              <div className="bg-zinc-950/40 p-4 rounded-xl border border-zinc-900/80 flex flex-col gap-4">
                <h4 className="text-xs font-bold text-zinc-400 flex items-center gap-1.5">
                  <Search className="w-3.5 h-3.5 text-zinc-400" />
                  Semantic Search Recall Test
                </h4>
                <form onSubmit={handleRecallTest} className="flex gap-2">
                  <input
                    type="text"
                    value={recallQuery}
                    onChange={(e) => setRecallQuery(e.target.value)}
                    placeholder="Search queries (e.g. JWT token authorization)..."
                    className="flex-1 bg-zinc-950/80 border border-zinc-800 rounded-xl px-3 py-2 text-xs text-zinc-300 placeholder-zinc-600 focus:outline-none focus:border-violet-500/50 font-medium"
                  />
                  <button
                    type="submit"
                    disabled={testingRecall || !recallQuery.trim()}
                    className="px-4 py-2 bg-zinc-900 hover:bg-zinc-800 border border-zinc-800 hover:border-zinc-700 text-xs font-semibold text-zinc-200 transition-all rounded-xl cursor-pointer"
                  >
                    {testingRecall ? 'Querying...' : 'Recall'}
                  </button>
                </form>

                {recallResults.length > 0 && (
                  <div className="flex flex-col gap-2 mt-2 border-t border-zinc-900/80 pt-3">
                    <div className="text-[10px] text-zinc-500 font-bold uppercase tracking-wider mb-1">
                      Recall Matches
                    </div>
                    <div className="flex flex-col gap-2 max-h-60 overflow-y-auto pr-1">
                      {recallResults.map(match => (
                        <div key={match.id} className="bg-zinc-900/20 border border-zinc-900/80 p-3 rounded-xl flex items-start justify-between gap-4">
                          <div className="flex flex-col gap-1.5">
                            <p className="text-xs font-medium text-zinc-300 leading-relaxed">{match.content}</p>
                            <div className="flex items-center gap-2 text-[10px] font-mono">
                              <span className="text-zinc-500">ID: {match.id}</span>
                              <span className="text-zinc-700">•</span>
                              <span className="text-zinc-400 font-bold">SCORE: {match.score.toFixed(4)}</span>
                              <span className="text-zinc-700">•</span>
                              <span className={`px-2 py-0.5 rounded font-bold uppercase ${getTrustBgColor(match.trust)}`}>
                                Trust: {match.trust.toFixed(2)}
                              </span>
                              <span className="text-zinc-700">•</span>
                              <span className={`px-2 py-0.5 rounded font-bold uppercase ${getStateBadgeColor(match.state)}`}>
                                {match.state}
                              </span>
                            </div>
                          </div>
                          <button
                            onClick={() => handleDeleteMemory(match.id)}
                            className="p-1 text-zinc-600 hover:text-red-400 transition-all cursor-pointer"
                            title="Forget"
                          >
                            <Trash2 className="w-3.5 h-3.5" />
                          </button>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>

              {/* All memories list */}
              <div className="flex flex-col gap-3">
                <h4 className="text-xs font-bold text-zinc-400 flex items-center gap-1.5">
                  <FileText className="w-3.5 h-3.5 text-zinc-400" />
                  Stored Memories Explorer ({selectedDb.memories.length})
                </h4>

                {selectedDb.memories.length === 0 ? (
                  <div className="text-zinc-600 text-xs py-8 text-center border border-dashed border-zinc-900 rounded-xl bg-zinc-950/20">
                    No memories found. Store some above or let your agent fill it!
                  </div>
                ) : (
                  <div className="flex flex-col gap-2.5 max-h-[450px] overflow-y-auto pr-1.5">
                    {selectedDb.memories.map(mem => (
                      <div key={mem.id} className="bg-zinc-900/10 hover:bg-zinc-900/20 border border-zinc-900/60 hover:border-zinc-800/80 p-4 rounded-xl transition-all flex items-start justify-between gap-4">
                        <div className="flex flex-col gap-2 flex-1">
                          <p className="text-xs font-medium text-zinc-300 leading-relaxed">{mem.content}</p>
                          <div className="flex flex-wrap items-center gap-3 text-[10px] font-mono">
                            <span className="text-zinc-500">ID: {mem.id}</span>
                            <span className="text-zinc-700">•</span>
                            <span className={`px-1.5 py-0.5 rounded font-semibold uppercase ${getStateBadgeColor(mem.state)}`}>
                              {mem.state}
                            </span>
                            <span className="text-zinc-700">•</span>
                            <span className="text-zinc-400 flex items-center gap-1">
                              <Shield className="w-3 h-3 text-zinc-500" />
                              Trust: {mem.trust.toFixed(2)}
                            </span>
                            <span className="text-zinc-700">•</span>
                            <span className="text-zinc-400 flex items-center gap-1">
                              <TrendingDown className="w-3 h-3 text-zinc-500" />
                              Uncertainty: {mem.uncertainty.toFixed(2)}
                            </span>
                            {mem.contradiction_group !== undefined && (
                              <>
                                <span className="text-zinc-700">•</span>
                                <span className="text-purple-400 bg-purple-950/30 px-1.5 py-0.5 rounded font-bold border border-purple-500/10">
                                  Group {mem.contradiction_group}
                                </span>
                              </>
                            )}
                          </div>
                        </div>

                        <button
                          onClick={() => handleDeleteMemory(mem.id)}
                          className="p-1.5 text-zinc-700 hover:text-red-400 hover:bg-zinc-950/40 rounded-lg transition-all cursor-pointer"
                          title="Forget"
                        >
                          <Trash2 className="w-3.5 h-3.5" />
                        </button>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Telemetry Timeline logs stream */}
          <div className="bg-zinc-900/30 backdrop-blur-md border border-zinc-800/80 rounded-2xl p-6 flex flex-col gap-5">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Terminal className="w-4.5 h-4.5 text-indigo-400" />
                <h3 className="font-bold text-sm text-zinc-200">Telemetry Log Stream</h3>
              </div>
              <span className="text-[10px] text-zinc-500 font-bold uppercase tracking-wider">
                mcp-server.jsonl
              </span>
            </div>

            {loadingLogs ? (
              <div className="text-zinc-500 text-xs py-4 flex items-center gap-2">
                <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                Loading logs...
              </div>
            ) : logs.length === 0 ? (
              <div className="text-zinc-600 text-xs py-8 text-center border border-dashed border-zinc-900 rounded-xl bg-zinc-950/20">
                No logs detected. Ensure the MCP server is configured and active.
              </div>
            ) : (
              <div className="flex flex-col gap-2 max-h-[300px] overflow-y-auto pr-1 text-xs font-mono">
                {logs.map((log, index) => (
                  <div 
                    key={index}
                    className={`border-l-2 p-3 rounded-r-xl transition-all ${getLogColor(log)}`}
                  >
                    <div className="flex items-center justify-between gap-4 mb-1">
                      <span className="text-[10px] font-bold px-1.5 py-0.5 rounded bg-zinc-950/50 uppercase tracking-wide">
                        {getLogBadge(log)}
                      </span>
                      <span className="text-[10px] text-zinc-500">
                        {new Date(log.timestamp).toLocaleTimeString()}
                      </span>
                    </div>

                    <p className="text-[11px] leading-relaxed break-all font-medium text-zinc-300">
                      {log.message}
                    </p>

                    {log.query && (
                      <div className="mt-1.5 text-[10px] bg-zinc-950/60 p-2 rounded border border-zinc-900 text-zinc-400">
                        <span className="text-zinc-600">QUERY: </span> &quot;{log.query}&quot;
                        {log.results && (
                          <div className="mt-1 flex flex-col gap-0.5">
                            <span className="text-zinc-600">RECALL MATCHES: </span>
                            {log.results.map((r: any, ri: number) => (
                              <div key={ri} className="text-zinc-500 pl-2">
                                - ID={r.id} Score={r.score.toFixed(3)}: &quot;{r.content.substring(0, 45)}...&quot;
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        </section>
      </main>

      {/* Footer */}
      <footer className="border-t border-zinc-900 bg-zinc-950/80 py-6 text-center text-xs text-zinc-600 mt-auto">
        <div className="max-w-7xl mx-auto px-6 flex flex-col md:flex-row items-center justify-between gap-4">
          <p>© 2026 Memoire AI. Open-source local-first semantic memory.</p>
          <div className="flex items-center gap-4">
            <span className="bg-zinc-900 text-zinc-500 border border-zinc-800 px-2 py-0.5 rounded font-mono text-[10px]">
              V0.1.0-PRODUCTION
            </span>
          </div>
        </div>
      </footer>
    </div>
  );
}
