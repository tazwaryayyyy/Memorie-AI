use memoire::Memoire;

fn main() -> anyhow::Result<()> {
    env_logger::init();

    let db = "agent_memory.db";
    let m = Memoire::new(db)?;

    println!("=== Memoire Basic Usage Demo ===\n");

    // Store memories
    let sessions = [
        "Session 2024-01-15: Fixed an off-by-one error in the user pagination \
         endpoint. The `limit` parameter was applied before `offset`, causing \
         the first page to return one fewer result than expected.",

        "Session 2024-01-16: Redis cache was not being invalidated after a user \
         profile update. Added cache eviction in `update_profile` service method.",

        "Session 2024-01-17: Refactored authentication middleware to use async \
         token validation. Reduced p99 latency on authenticated routes from 340ms \
         to 45ms.",

        "Session 2024-01-18: Diagnosed a memory leak in the websocket connection \
         handler. Connections were not being dropped on client disconnect. Fixed \
         by implementing the Drop trait on the handler struct.",

        "Session 2024-01-19: Upgraded the ORM from v1 to v2. Required migration \
         of 14 raw SQL queries to the new query builder API. All existing tests pass.",
    ];

    println!("Storing {} sessions...", sessions.len());
    for s in &sessions {
        let ids = m.remember(s)?;
        println!("  stored {} chunk(s): {}...", ids.len(), &s[..60]);
    }
    println!("\nTotal chunks in store: {}\n", m.count()?);

    // Query
    let queries = [
        "how did I improve response time?",
        "what cache or Redis issues did I fix?",
        "memory leaks and resource management",
        "pagination bugs",
    ];

    for q in &queries {
        println!("Query: \"{q}\"");
        let results = m.recall(q, 2)?;
        for (i, r) in results.iter().enumerate() {
            println!("  [{i}] score={:.4}  {}", r.score, &r.content[..80.min(r.content.len())]);
        }
        println!();
    }

    Ok(())
}
