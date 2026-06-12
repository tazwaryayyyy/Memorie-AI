//! Memoire CLI
//!
//! A command-line interface to the Memoire memory engine.
//!
//! Usage:
//!   memoire --db ./agent.db remember "Fixed the JWT bug today"
//!   memoire --db ./agent.db recall "authentication issues" --top 5
//!   memoire --db ./agent.db recall "auth" --json
//!   memoire --db ./agent.db forget 42
//!   memoire --db ./agent.db count
//!   memoire --db ./agent.db clear --confirm
//!   memoire --db ./agent.db import ./notes.txt
//!   memoire --db ./agent.db export

use std::env;
use std::fs;
use std::io::{self, BufRead, Write};
use std::process;

use memoire::Memoire;

const HELP: &str = r#"
Memoire — local-first semantic memory for AI agents

USAGE:
    memoire [OPTIONS] <COMMAND> [ARGS]

OPTIONS:
    --db <PATH>     Path to the SQLite database (default: ./memoire.db)
    --help, -h      Show this help message

COMMANDS:
    remember <TEXT>               Store TEXT as a memory (reads stdin if TEXT is "-")
    recall   <QUERY> [--top N]    Find the N most similar memories (default: 5)
                     [--json]     Output raw JSON
                     [--min-score F] Only return results above score threshold
    forget   <ID>                 Delete memory by id
    count                         Print total stored memory chunks
    clear    [--confirm]          Erase ALL memories (requires --confirm flag)
    import   <FILE>               Import memories line-by-line from a file
    export   [--json]             Dump all memories to stdout
    info                          Show database info and stats
    cache-models                  Pre-download embedding models to local cache

EXAMPLES:
    memoire remember "Fixed off-by-one error in pagination endpoint"
    memoire recall "what pagination bugs did I fix?" --top 3
    memoire recall "auth" --json | jq '.[0].content'
    memoire recall "performance" --min-score 0.7
    echo "Refactored DB pool" | memoire remember -
    memoire import ./session_notes.txt
    memoire export --json > backup.json
    memoire --db /tmp/test.db count
"#;

struct Args {
    db: String,
    command: Command,
}

enum Command {
    Remember {
        text: String,
    },
    Recall {
        query: String,
        top: usize,
        json: bool,
        min_score: f32,
    },
    Forget {
        id: i64,
    },
    Count,
    Clear {
        confirmed: bool,
    },
    Import {
        path: String,
    },
    Export {
        json: bool,
    },
    ExportNamespace {
        db_path: String,
        namespace: Option<String>,
        output: Option<String>,
    },
    ImportNamespace {
        db_path: String,
        file_path: String,
        namespace: Option<String>,
    },
    Info,
    CacheModels,
    Help,
}

fn parse_args() -> Result<Args, String> {
    let raw: Vec<String> = env::args().skip(1).collect();
    let mut db = "./memoire.db".to_string();
    let mut i = 0;

    // Parse global flags
    while i < raw.len() {
        match raw[i].as_str() {
            "--db" => {
                i += 1;
                db = raw.get(i).cloned().ok_or("--db requires a path")?;
            }
            "--help" | "-h" => {
                return Ok(Args {
                    db,
                    command: Command::Help,
                });
            }
            _ => break,
        }
        i += 1;
    }

    let subcmd = raw.get(i).map(String::as_str).unwrap_or("");
    i += 1;

    let command = match subcmd {
        "remember" => {
            let text = raw.get(i).cloned().unwrap_or_default();
            if text.is_empty() {
                return Err("remember requires text (or \"-\" to read stdin)".into());
            }
            let text = if text == "-" {
                let stdin = io::stdin();
                stdin
                    .lock()
                    .lines()
                    .map_while(|l| l.ok())
                    .collect::<Vec<_>>()
                    .join("\n")
            } else {
                raw[i..].join(" ")
            };
            Command::Remember { text }
        }

        "recall" => {
            let query = raw.get(i).cloned().ok_or("recall requires a query")?;
            i += 1;
            let mut top = 5usize;
            let mut json = false;
            let mut min_score = 0.0f32;
            while i < raw.len() {
                match raw[i].as_str() {
                    "--top" | "-n" => {
                        i += 1;
                        top = raw
                            .get(i)
                            .and_then(|s| s.parse().ok())
                            .ok_or("--top requires a number")?;
                    }
                    "--json" => json = true,
                    "--min-score" => {
                        i += 1;
                        min_score = raw
                            .get(i)
                            .and_then(|s| s.parse().ok())
                            .ok_or("--min-score requires a float")?;
                    }
                    _ => {}
                }
                i += 1;
            }
            Command::Recall {
                query,
                top,
                json,
                min_score,
            }
        }

        "forget" => {
            let id: i64 = raw
                .get(i)
                .and_then(|s| s.parse().ok())
                .ok_or("forget requires an integer id")?;
            Command::Forget { id }
        }

        "count" => Command::Count,

        "clear" => {
            let confirmed = raw.get(i).map(String::as_str) == Some("--confirm");
            Command::Clear { confirmed }
        }

        "import" => {
            let mut pos_args = Vec::new();
            let mut temp_i = i;
            while temp_i < raw.len() {
                if temp_i < raw.len() && !raw[temp_i].starts_with('-') {
                    pos_args.push(raw[temp_i].clone());
                } else if temp_i < raw.len() && raw[temp_i] == "--namespace" {
                    temp_i += 1; // skip next
                }
                temp_i += 1;
            }

            if pos_args.len() >= 2 {
                let db_path = raw.get(i).cloned().unwrap();
                i += 1;
                let file_path = raw.get(i).cloned().unwrap();
                i += 1;
                let mut namespace = None;
                while i < raw.len() {
                    if raw[i] == "--namespace" {
                        i += 1;
                        namespace = raw.get(i).cloned();
                    }
                    i += 1;
                }
                Command::ImportNamespace { db_path, file_path, namespace }
            } else {
                let path = raw.get(i).cloned().ok_or("import requires a file path")?;
                Command::Import { path }
            }
        }

        "export" => {
            let next_arg = raw.get(i).map(String::as_str);
            if next_arg == Some("--json") || next_arg.is_none() {
                let json = next_arg == Some("--json");
                Command::Export { json }
            } else {
                let db_path = raw.get(i).cloned().ok_or("export requires a database path")?;
                i += 1;
                let mut namespace = None;
                let mut output = None;
                while i < raw.len() {
                    match raw[i].as_str() {
                        "--namespace" => {
                            i += 1;
                            namespace = raw.get(i).cloned();
                        }
                        "--output" => {
                            i += 1;
                            output = raw.get(i).cloned();
                        }
                        _ => {}
                    }
                    i += 1;
                }
                Command::ExportNamespace { db_path, namespace, output }
            }
        }

        "info" => Command::Info,
        "cache-models" => Command::CacheModels,

        "" | "--help" | "-h" => Command::Help,

        other => return Err(format!("unknown command: {other:?}. Run with --help.")),
    };

    Ok(Args { db, command })
}

fn run() -> anyhow::Result<()> {
    let _ = env_logger::try_init();

    let args = parse_args().unwrap_or_else(|e| {
        eprintln!("error: {e}");
        process::exit(1);
    });

    if let Command::Help = args.command {
        print!("{HELP}");
        return Ok(());
    }

    if let Command::CacheModels = args.command {
        println!("Pre-downloading embedding models to cache...");
        let _ = memoire::embedder::Embedder::new()?;
        println!("✓ Model caching complete. You can now use Memoire in offline mode.");
        return Ok(());
    }

    let m = Memoire::new(&args.db)?;

    match args.command {
        Command::Help => unreachable!(),

        Command::Remember { text } => {
            if text.trim().is_empty() {
                eprintln!("error: input is empty, nothing stored.");
                process::exit(1);
            }
            let ids = m.remember(&text)?;
            println!("✓ stored {} chunk(s) → ids: {:?}", ids.len(), ids);
        }

        Command::Recall {
            query,
            top,
            json,
            min_score,
        } => {
            let mut results = m.recall(&query, top)?;
            results.retain(|r| r.score >= min_score);

            if results.is_empty() {
                if json {
                    println!("[]");
                } else {
                    println!("(no results)");
                }
                return Ok(());
            }

            if json {
                println!("{}", serde_json::to_string_pretty(&results)?);
            } else {
                println!();
                for (i, r) in results.iter().enumerate() {
                    let bar = score_bar(r.score, 20);
                    println!("  [{i}] id={:<6} score={:.4}  {bar}", r.id, r.score);
                    // Word-wrap content at 72 chars
                    for line in wrap(&r.content, 72) {
                        println!("       {line}");
                    }
                    println!();
                }
            }
        }

        Command::Forget { id } => {
            if m.forget(id)? {
                println!("✓ deleted memory id={id}");
            } else {
                eprintln!("✗ no memory found with id={id}");
                process::exit(1);
            }
        }

        Command::Count => {
            println!("{}", m.count()?);
        }

        Command::Clear { confirmed } => {
            if !confirmed {
                eprintln!("⚠ This will erase ALL memories. Add --confirm to proceed.");
                process::exit(1);
            }
            let n = m.count()?;
            m.clear()?;
            println!("✓ cleared {n} memory chunk(s).");
        }

        Command::Import { path } => {
            let content = fs::read_to_string(&path)
                .map_err(|e| anyhow::anyhow!("cannot read {path}: {e}"))?;

            let lines: Vec<&str> = content
                .lines()
                .map(str::trim)
                .filter(|l| !l.is_empty() && !l.starts_with('#'))
                .collect();

            if lines.is_empty() {
                eprintln!("file is empty or contains only comments.");
                return Ok(());
            }

            println!("importing {} line(s) from {path}...", lines.len());
            let mut total_chunks = 0usize;
            for (i, line) in lines.iter().enumerate() {
                let ids = m.remember(line)?;
                total_chunks += ids.len();
                print!(
                    "\r  [{}/{}] {} chunk(s) stored",
                    i + 1,
                    lines.len(),
                    total_chunks
                );
                io::stdout().flush()?;
            }
            println!("\n✓ done. {total_chunks} total chunk(s) stored.");
        }

        Command::Export { json } => {
            let results = m.export_all()?;
            if results.is_empty() {
                eprintln!("(no memories stored)");
                return Ok(());
            }
            if json {
                println!("{}", serde_json::to_string_pretty(&results)?);
            } else {
                for r in &results {
                    println!("[id={}  {}] {}", r.id, r.created_at, r.content);
                }
            }
        }

        Command::ExportNamespace { db_path, namespace, output } => {
            let ns = namespace.unwrap_or_else(|| "default".to_string());
            let m = Memoire::new_ns(&db_path, &ns)?;
            let snapshot = m.export_namespace()?;
            let json_str = serde_json::to_string_pretty(&snapshot)?;
            if let Some(out_path) = output {
                fs::write(&out_path, json_str)
                    .map_err(|e| anyhow::anyhow!("failed to write output file: {e}"))?;
            } else {
                println!("{json_str}");
            }
        }

        Command::ImportNamespace { db_path, file_path, namespace } => {
            let json_str = if file_path == "-" {
                let stdin = io::stdin();
                let mut input = String::new();
                stdin.lock().read_to_string(&mut input)
                    .map_err(|e| anyhow::anyhow!("failed to read stdin: {e}"))?;
                input
            } else {
                fs::read_to_string(&file_path)
                    .map_err(|e| anyhow::anyhow!("failed to read file: {e}"))?
            };
            let snapshot: serde_json::Value = serde_json::from_str(&json_str)
                .map_err(|e| anyhow::anyhow!("failed to parse import JSON: {e}"))?;
                
            let ns = namespace
                .or_else(|| snapshot.get("namespace").and_then(|v| v.as_str()).map(String::from))
                .unwrap_or_else(|| "default".to_string());
                
            let m = Memoire::new_ns(&db_path, &ns)?;
            let count = m.import_namespace(&snapshot)?;
            println!("✓ imported {count} memories into namespace '{ns}'");
        }

        Command::Info => {
            let count = m.count()?;
            let db_path = std::fs::canonicalize(&args.db)
                .map(|p| p.display().to_string())
                .unwrap_or_else(|_| args.db.clone());
            let db_size = fs::metadata(&args.db)
                .map(|m| format!("{:.1} KB", m.len() as f64 / 1024.0))
                .unwrap_or_else(|_| "unknown".into());

            println!();
            println!("  Memoire Database Info");
            println!("  ─────────────────────────────────────");
            println!("  Path:    {db_path}");
            println!("  Size:    {db_size}");
            println!("  Chunks:  {count}");
            println!("  Model:   all-MiniLM-L6-v2 (384-dim)");
            println!("  Engine:  fastembed + rusqlite (bundled SQLite)");
            println!();
        }

        // Handled above via early return before Memoire::new() is called
        Command::CacheModels => unreachable!(),
    }

    Ok(())
}

fn score_bar(score: f32, width: usize) -> String {
    let filled = (score * width as f32).round() as usize;
    let empty = width.saturating_sub(filled);
    format!("[{}{}]", "█".repeat(filled), "░".repeat(empty))
}

fn wrap(s: &str, width: usize) -> Vec<String> {
    let mut lines = Vec::new();
    let mut line = String::new();
    for word in s.split_whitespace() {
        if !line.is_empty() && line.len() + word.len() + 1 > width {
            lines.push(line.clone());
            line.clear();
        }
        if !line.is_empty() {
            line.push(' ');
        }
        line.push_str(word);
    }
    if !line.is_empty() {
        lines.push(line);
    }
    lines
}

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        process::exit(1);
    }
}
