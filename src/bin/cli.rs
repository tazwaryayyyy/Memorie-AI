//! Memoire CLI
//!
//! A command-line interface to the Memoire memory engine.
//!
//! Usage:
//!   memoire [--db PATH] [--ns NAMESPACE] <COMMAND> [ARGS]
//!
//! Run `memoire --help` for the full command reference.

use std::env;
use std::fs;
use std::io::{self, BufRead, Read, Write};
use std::process;

use memoire::Memoire;

const HELP: &str = r#"
Memoire — local-first semantic memory for AI agents

USAGE:
    memoire [OPTIONS] <COMMAND> [ARGS]

OPTIONS:
    --db <PATH>          Path to the SQLite database (default: ./memoire.db)
    --ns <NAMESPACE>     Namespace within the database (default: default)
    --help, -h           Show this help message

COMMANDS:
    remember <TEXT>               Store TEXT as a memory (use "-" to read stdin)
    recall   <QUERY>              Find the most similar memories
             [--top N]            Return up to N results (default: 5)
             [--json]             Output raw JSON
             [--min-score F]      Only return results above score threshold
    forget   <ID>                 Delete memory by id
    count                         Print total stored memory chunks
    clear    [--confirm]          Erase ALL memories in the namespace (requires --confirm)
    export   [--output FILE]      Export namespace snapshot to JSON (stdout or file)
    import   <FILE>               Import a JSON snapshot (use "-" to read stdin)
    ingest   <FILE>               Ingest memories line-by-line from a plain-text file
    info                          Show database info and stats
    cache-models                  Pre-download embedding models to local cache

EXAMPLES:
    memoire remember "Fixed off-by-one error in pagination endpoint"
    memoire recall "what pagination bugs did I fix?" --top 3
    memoire recall "auth" --json | jq '.[0].content'
    memoire recall "performance" --min-score 0.7
    echo "Refactored DB pool" | memoire remember -
    memoire export --output backup.json
    memoire --ns billing-agent export --output billing.json
    memoire --ns billing-agent import billing.json
    memoire ingest ./session_notes.txt
    memoire --db /tmp/test.db count
"#;

struct Args {
    db: String,
    ns: String,
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
    /// Export the current namespace as a JSON snapshot.
    Export {
        output: Option<String>,
    },
    /// Import a JSON snapshot into the current namespace.
    Import {
        file: String,
    },
    /// Ingest memories line-by-line from a plain-text file (legacy import).
    Ingest {
        path: String,
    },
    Info,
    CacheModels,
    Help,
}

fn parse_args() -> Result<Args, String> {
    let raw: Vec<String> = env::args().skip(1).collect();
    let mut db = "./memoire.db".to_string();
    let mut ns = "default".to_string();
    let mut i = 0;

    // Parse global flags before the subcommand.
    while i < raw.len() {
        match raw[i].as_str() {
            "--db" => {
                i += 1;
                db = raw.get(i).cloned().ok_or("--db requires a path")?;
            }
            "--ns" | "--namespace" => {
                i += 1;
                ns = raw.get(i).cloned().ok_or("--ns requires a namespace")?;
            }
            "--help" | "-h" => {
                return Ok(Args {
                    db,
                    ns,
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
                io::stdin()
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
                            .ok_or("--top requires a positive integer")?;
                    }
                    "--json" => json = true,
                    "--min-score" => {
                        i += 1;
                        min_score = raw
                            .get(i)
                            .and_then(|s| s.parse().ok())
                            .ok_or("--min-score requires a float in [0.0, 1.0]")?;
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
                .ok_or("forget requires an integer memory id")?;
            Command::Forget { id }
        }

        "count" => Command::Count,

        "clear" => {
            let confirmed = raw.get(i).map(String::as_str) == Some("--confirm");
            Command::Clear { confirmed }
        }

        "export" => {
            let mut output = None;
            while i < raw.len() {
                if raw[i] == "--output" || raw[i] == "-o" {
                    i += 1;
                    output = raw.get(i).cloned();
                }
                i += 1;
            }
            Command::Export { output }
        }

        "import" => {
            let file = raw
                .get(i)
                .cloned()
                .ok_or("import requires a file path (or \"-\" for stdin)")?;
            Command::Import { file }
        }

        "ingest" => {
            let path = raw.get(i).cloned().ok_or("ingest requires a file path")?;
            Command::Ingest { path }
        }

        "info" => Command::Info,
        "cache-models" => Command::CacheModels,

        "" | "--help" | "-h" => Command::Help,

        other => return Err(format!("unknown command: {other:?}. Run with --help.")),
    };

    Ok(Args { db, ns, command })
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

    let m = Memoire::new_ns(&args.db, &args.ns)?;

    match args.command {
        Command::Help | Command::CacheModels => unreachable!(),

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
                for (idx, r) in results.iter().enumerate() {
                    let bar = score_bar(r.score, 20);
                    println!("  [{idx}] id={:<6} score={:.4}  {bar}", r.id, r.score);
                    for line in word_wrap(&r.content, 72) {
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
                eprintln!(
                    "⚠ This will erase ALL memories in namespace '{}'. Add --confirm to proceed.",
                    args.ns
                );
                process::exit(1);
            }
            let n = m.count()?;
            m.clear()?;
            println!(
                "✓ cleared {n} memory chunk(s) from namespace '{}'.",
                args.ns
            );
        }

        Command::Export { output } => {
            let snapshot = m.export_namespace()?;
            let json_str = serde_json::to_string_pretty(&snapshot)?;
            match output {
                Some(ref path) => {
                    fs::write(path, &json_str)
                        .map_err(|e| anyhow::anyhow!("failed to write {path}: {e}"))?;
                    println!("✓ exported to {path}");
                }
                None => println!("{json_str}"),
            }
        }

        Command::Import { file } => {
            let json_str = if file == "-" {
                let mut buf = String::new();
                io::stdin()
                    .lock()
                    .read_to_string(&mut buf)
                    .map_err(|e| anyhow::anyhow!("failed to read stdin: {e}"))?;
                buf
            } else {
                fs::read_to_string(&file)
                    .map_err(|e| anyhow::anyhow!("failed to read {file}: {e}"))?
            };

            let snapshot: serde_json::Value = serde_json::from_str(&json_str)
                .map_err(|e| anyhow::anyhow!("invalid JSON snapshot: {e}"))?;

            let count = m.import_namespace(&snapshot)?;
            println!("✓ imported {count} memories into namespace '{}'", args.ns);
        }

        Command::Ingest { path } => {
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

            println!("ingesting {} line(s) from {path}...", lines.len());
            let mut total_chunks = 0usize;
            for (idx, line) in lines.iter().enumerate() {
                let ids = m.remember(line)?;
                total_chunks += ids.len();
                print!(
                    "\r  [{}/{}] {} chunk(s) stored",
                    idx + 1,
                    lines.len(),
                    total_chunks
                );
                io::stdout().flush()?;
            }
            println!("\n✓ done. {total_chunks} total chunk(s) stored.");
        }

        Command::Info => {
            let count = m.count()?;
            let db_path = std::fs::canonicalize(&args.db)
                .map(|p| p.display().to_string())
                .unwrap_or_else(|_| args.db.clone());
            let db_size = fs::metadata(&args.db)
                .map(|meta| format!("{:.1} KB", meta.len() as f64 / 1024.0))
                .unwrap_or_else(|_| "unknown".into());

            println!();
            println!("  Memoire Database Info");
            println!("  ─────────────────────────────────────");
            println!("  Path:       {db_path}");
            println!("  Namespace:  {}", args.ns);
            println!("  Size:       {db_size}");
            println!("  Chunks:     {count}");
            println!("  Model:      all-MiniLM-L6-v2 (384-dim)");
            println!("  Engine:     fastembed + rusqlite (bundled SQLite)");
            println!();
        }
    }

    Ok(())
}

fn score_bar(score: f32, width: usize) -> String {
    let filled = (score * width as f32).round() as usize;
    let empty = width.saturating_sub(filled);
    format!("[{}{}]", "█".repeat(filled), "░".repeat(empty))
}

fn word_wrap(s: &str, width: usize) -> Vec<String> {
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
