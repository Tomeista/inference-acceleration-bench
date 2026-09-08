"""Offline seed material for the frozen prompt sets.

Why synthetic rather than ShareGPT/LongBench/BFCL: the prompt sets have to be
buildable and reviewable without network access or a HuggingFace token, and for
the BF16-vs-FP8 comparison the *content* of a prompt is irrelevant. Only its
length matters, and length here is exact.

This stops being true for speculative decoding. Acceptance rate is a function of
how predictable the model finds its own continuation, so before the EAGLE-3
phase these builders must be pointed at real data. Until then, treat any
acceptance number measured on this corpus as an indication, not a result.
"""

from __future__ import annotations

SHORT_QUESTIONS = [
    "What is the difference between a process and a thread?",
    "How does a bloom filter avoid false negatives?",
    "Explain why floating point addition is not associative.",
    "What problem does a write-ahead log solve?",
    "When would you choose a columnar storage format over a row store?",
    "Why is DNS resolution often the slowest part of a cold HTTP request?",
    "What is the purpose of a memory barrier?",
    "How does copy-on-write make fork cheap?",
    "Describe the tradeoff between recall and precision.",
    "Why do hash maps degrade to linear time in the worst case?",
    "What does the CAP theorem actually constrain?",
    "How does TCP slow start interact with short-lived connections?",
    "Explain what makes a function tail recursive.",
    "Why is UTF-8 self-synchronizing?",
    "What is the difference between latency and throughput?",
    "How does a garbage collector identify unreachable objects?",
    "Why do B-trees suit disk storage better than binary search trees?",
    "What is a race condition and how is it different from a deadlock?",
    "Explain the role of a scheduler quantum in preemptive multitasking.",
    "How does content addressing differ from location addressing?",
    "What makes a cryptographic hash function preimage resistant?",
    "Why can a cache line be a source of false sharing?",
    "What is the difference between a mutex and a semaphore?",
    "How does a reverse proxy differ from a forward proxy?",
    "Explain why quicksort is usually faster than mergesort in practice.",
    "What is backpressure in a streaming system?",
    "How does a vector clock detect concurrent updates?",
    "Why is idempotency important for retry logic?",
    "What does a JIT compiler do that an interpreter cannot?",
    "Explain the difference between authentication and authorization.",
    "How does branch prediction affect tight loop performance?",
    "What is the purpose of a nonce in a network protocol?",
    "Why do distributed systems need a consensus protocol?",
    "What is the difference between eventual and strong consistency?",
    "How does a skip list achieve logarithmic search?",
    "Explain why premature optimization is often counterproductive.",
    "What is the role of an inode in a filesystem?",
    "How does connection pooling reduce database load?",
    "Why is it hard to measure the latency of a single request accurately?",
    "What does it mean for a data structure to be lock-free?",
]

LONGFORM_INSTRUCTIONS = [
    "Write a detailed technical overview of how modern garbage collectors balance throughput against pause time.",
    "Explain, at length and for an engineering audience, how a distributed log guarantees ordering across partitions.",
    "Write an essay on the design tradeoffs between microservices and a well-factored monolith.",
    "Produce a thorough explanation of how virtual memory paging works, from the page table down to the TLB.",
    "Write a comprehensive guide to debugging intermittent failures in a distributed system.",
    "Explain in depth how a query planner decides between a hash join and a nested loop join.",
    "Write a long-form article about the history and design of the Unix filesystem abstraction.",
    "Describe thoroughly how TLS establishes a session, including the role of each handshake message.",
    "Write a detailed comparison of optimistic and pessimistic concurrency control.",
    "Explain at length why cache invalidation is considered one of the hard problems in computing.",
    "Write an extended technical piece on how compilers perform escape analysis and why it matters.",
    "Produce a thorough discussion of the tradeoffs involved in choosing a data serialization format.",
    "Write a detailed account of how a modern CPU executes instructions out of order while preserving semantics.",
    "Explain comprehensively how rate limiting is implemented at scale and where naive approaches fail.",
    "Write a long technical overview of consensus algorithms, contrasting Paxos with Raft.",
    "Describe in detail how a container runtime isolates a process using namespaces and cgroups.",
    "Write an in-depth explanation of how gradient descent variants differ in practice.",
    "Produce a detailed article on the engineering behind low-latency network file systems.",
    "Explain thoroughly how a build system decides what needs rebuilding and why correctness is hard.",
    "Write a comprehensive piece on observability, contrasting metrics, logs, and distributed traces.",
    "Describe at length how a key-value store implements compaction and why it causes latency spikes.",
    "Write a detailed technical essay on the tradeoffs of schema-on-read versus schema-on-write.",
    "Explain in depth how modern search engines rank documents and how relevance is evaluated.",
    "Produce a long-form explanation of how a load balancer maintains session affinity.",
    "Write a thorough overview of memory allocators and why general-purpose allocation is difficult.",
    "Explain comprehensively how time synchronization works across a fleet of machines.",
    "Write a detailed technical guide to designing an idempotent API for financial transactions.",
    "Describe at length how a profiler attributes cost to a line of code and where it misleads.",
    "Write an extended piece on the engineering constraints of running stateful services in a cluster scheduler.",
    "Explain in detail how a modern database implements multi-version concurrency control.",
]

# Neutral technical prose, used both to pad a prompt to an exact token budget
# and to synthesize the long documents for the RAG class.
DOC_PARAGRAPHS = [
    "The storage layer maintains an append-only segment file for each partition. Writes are buffered in memory and flushed once the segment reaches its configured size, at which point the segment is sealed and becomes immutable. Immutability simplifies replication considerably, because a sealed segment can be copied byte for byte without coordination.",
    "Compaction runs as a background process and merges overlapping segments into a smaller set of files. The scheduler prioritizes segments whose key ranges overlap most heavily, since those contribute the largest read amplification. Operators can throttle compaction throughput, though doing so trades a lower steady-state latency for a slower recovery after a write burst.",
    "Replication is quorum based. A write is acknowledged once a majority of replicas have durably recorded it, which bounds the number of failures the cluster tolerates without losing an acknowledged write. Read requests may be served by any replica, but a client requesting linearizable reads is routed to the current leader.",
    "The leader is elected using a randomized timeout. Each follower waits for a heartbeat, and if none arrives within its timeout it increments the term and campaigns for votes. Randomization makes split votes rare in practice, and the term number ensures that a stale leader returning from a network partition steps down immediately.",
    "Query planning begins with a logical rewrite phase that pushes predicates as close to the scan as possible. The planner then enumerates physical alternatives for each logical operator and selects a plan using estimated cardinalities. Estimation error compounds multiplicatively across joins, which is why plans for deeply nested queries are frequently poor.",
    "Statistics are collected by sampling rather than by full scan. The sampler maintains a reservoir per column and periodically recomputes histograms from it. Columns with heavy skew are given more buckets, because a uniform histogram would otherwise place most of the distribution into a single bucket and defeat the purpose.",
    "The cache is organized as a set-associative structure with a pseudo least recently used replacement policy. True LRU would require maintaining a total order within each set, which is expensive in hardware, so the policy approximates it using a tree of one-bit hints. The approximation is close enough that measured hit rates differ by under a percentage point.",
    "Memory is reclaimed using a generational collector. Most objects die young, so the collector scans the young generation frequently and the old generation rarely. Objects that survive several young collections are promoted, and a write barrier records references from old objects to young ones so that the young collection remains sound without scanning the entire heap.",
    "The scheduler assigns each runnable task a virtual runtime that advances in proportion to actual execution time divided by the weight of the task. Selecting the task with the smallest virtual runtime approximates fair sharing without maintaining explicit time slices. Interactive tasks benefit because they sleep often and therefore accumulate virtual runtime slowly.",
    "Network transfers are chunked and pipelined. The sender does not wait for an acknowledgment before dispatching the next chunk, so throughput is limited by the bandwidth-delay product rather than by round-trip time. The receiver advertises a window that shrinks under memory pressure, which provides backpressure without an explicit control channel.",
    "Failures are detected by a gossip protocol rather than by a central monitor. Each node periodically probes a randomly chosen peer and disseminates its observations. Detection latency grows logarithmically with cluster size, and the protocol tolerates message loss because a single missed probe does not by itself mark a node as failed.",
    "The index is a log-structured merge tree with a memory-resident top level. Lookups check the memory table first, then consult a bloom filter for each on-disk level to avoid unnecessary reads. A false positive costs one wasted seek, so the filters are sized to keep the false positive rate near one percent.",
    "Transactions use snapshot isolation implemented with multi-version storage. Each row carries the transaction identifier that created it and the one that superseded it, and a reader observes only versions committed before its snapshot began. Write conflicts are detected at commit time, and the later transaction aborts.",
    "The serialization format encodes field numbers rather than field names, so renaming a field is backward compatible while renumbering is not. Unknown fields are preserved on round trip, which allows an intermediate service to forward a message it does not fully understand without discarding information the eventual consumer needs.",
    "Autoscaling reacts to a smoothed load signal rather than to instantaneous utilization. Smoothing prevents the controller from oscillating in response to transient spikes, at the cost of a slower response to genuine sustained growth. The cooldown period after a scale-up is longer than after a scale-down for the same reason.",
    "Requests carry a trace identifier propagated through every downstream call. Each service emits spans annotated with timing and metadata, and a collector reassembles them into a tree. Sampling is decided at the entry point so that a trace is either complete or absent, since partially sampled traces are misleading when analyzed in aggregate.",
    "Authentication uses short-lived tokens issued by a central authority and verified locally against a published key. Local verification removes the authority from the request path, which matters because it would otherwise become a single point of failure. Revocation is therefore bounded by the token lifetime rather than being immediate.",
    "The build graph is derived from declared inputs rather than inferred from timestamps. Each action is keyed by a hash of its inputs and its command line, so a cache hit is safe even across machines. Undeclared inputs are the usual source of incorrect incremental builds, which is why the sandbox denies access to anything not declared.",
    "Batch jobs are checkpointed at operator boundaries. On failure the framework restarts from the most recent checkpoint rather than from the beginning, which bounds the wasted work. Checkpoint frequency is a tunable tradeoff, since writing a checkpoint consumes bandwidth that would otherwise serve the job itself.",
    "The allocator maintains size-class free lists per thread to avoid contention on a global lock. Blocks are returned to a central heap only when the cache of a thread exceeds a threshold, which keeps the common allocation path free of atomic operations. Fragmentation is bounded by rounding allocations up to the nearest size class.",
    "Model weights are sharded across devices along the hidden dimension, and activations are exchanged with a collective operation at each layer boundary. Communication overlaps with computation where the dependency structure permits, so the effective cost of sharding is well below the raw transfer time.",
    "Inference batches requests dynamically rather than on a fixed schedule. A request joins the current batch if it arrives before the batch is dispatched, so tail latency under light load stays close to the unbatched case while throughput under heavy load approaches the hardware limit.",
    "Attention keys and values are cached per sequence so that generating a token does not recompute the entire prefix. The cache grows linearly with sequence length and dominates memory consumption at long context, which is why paged allocation of the cache materially increases the number of concurrent sequences a device can hold.",
    "Quantized weights are dequantized inside the matrix multiplication kernel rather than ahead of time. This trades arithmetic for memory bandwidth, which is a favorable trade when the operation is bandwidth bound and an unfavorable one when it is compute bound. The crossover point moves with batch size.",
]

DOC_QUESTIONS = [
    "According to the document, what triggers a segment to be sealed?",
    "What does the document say determines how long revocation takes?",
    "Summarize what the document states about how failures are detected.",
    "Based on the text, why does estimation error matter for deeply nested queries?",
    "What reason does the document give for approximating LRU rather than implementing it exactly?",
    "According to the passage, what bounds the amount of work lost on failure?",
    "What does the document identify as the usual source of incorrect incremental builds?",
    "Per the text, what happens to the later transaction when a write conflict is detected?",
    "What does the document say about when quantization is a favorable trade?",
    "Explain what the passage says about why sampling is decided at the entry point.",
    "According to the document, how is contention on the allocator avoided?",
    "What reason does the text give for randomization making split votes rare?",
    "Based on the document, what limits transfer throughput?",
    "What does the passage say dominates memory consumption at long context?",
    "According to the text, why is the cooldown after a scale-up longer?",
    "What does the document state about renaming versus renumbering a field?",
    "Per the passage, what is the cost of a bloom filter false positive?",
    "What does the document say about how interactive tasks benefit from the scheduler?",
    "According to the text, why is immutability useful for replication?",
    "What does the passage say about how columns with heavy skew are handled?",
]

# Tool schemas for the structured-output class. Deliberately varied in arity and
# parameter type: argument-type fidelity is a metric the wider study cares
# about, so the prompt set should exercise more than string arguments.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_flights",
            "description": "Search available flights between two airports on a date.",
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {"type": "string", "description": "IATA code, e.g. FRA"},
                    "destination": {"type": "string", "description": "IATA code, e.g. LHR"},
                    "date": {"type": "string", "description": "ISO 8601 date"},
                    "max_stops": {"type": "integer", "description": "Maximum layovers"},
                    "refundable_only": {"type": "boolean"},
                },
                "required": ["origin", "destination", "date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_calendar_event",
            "description": "Create an event on the calendar of the user.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "start": {"type": "string", "description": "ISO 8601 datetime"},
                    "duration_minutes": {"type": "integer"},
                    "attendees": {"type": "array", "items": {"type": "string"}},
                    "visibility": {"type": "string", "enum": ["public", "private"]},
                },
                "required": ["title", "start", "duration_minutes"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_metrics",
            "description": "Query a time series metric over a window.",
            "parameters": {
                "type": "object",
                "properties": {
                    "metric": {"type": "string"},
                    "aggregation": {
                        "type": "string",
                        "enum": ["avg", "p50", "p95", "p99", "max"],
                    },
                    "window_minutes": {"type": "integer"},
                    "filters": {"type": "object", "description": "Label filters"},
                },
                "required": ["metric", "aggregation", "window_minutes"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "convert_currency",
            "description": "Convert an amount between two currencies.",
            "parameters": {
                "type": "object",
                "properties": {
                    "amount": {"type": "number"},
                    "from_currency": {"type": "string"},
                    "to_currency": {"type": "string"},
                },
                "required": ["amount", "from_currency", "to_currency"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "Send an email on behalf of the user.",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "array", "items": {"type": "string"}},
                    "subject": {"type": "string"},
                    "body": {"type": "string"},
                    "priority": {"type": "string", "enum": ["low", "normal", "high"]},
                },
                "required": ["to", "subject", "body"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather_forecast",
            "description": "Retrieve a weather forecast for a location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "days_ahead": {"type": "integer"},
                    "units": {"type": "string", "enum": ["metric", "imperial"]},
                },
                "required": ["city", "days_ahead"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_sql_query",
            "description": "Execute a read-only SQL query against the analytics warehouse.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer"},
                    "timeout_seconds": {"type": "number"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_ticket",
            "description": "Open a ticket in the issue tracker.",
            "parameters": {
                "type": "object",
                "properties": {
                    "project": {"type": "string"},
                    "summary": {"type": "string"},
                    "severity": {"type": "string", "enum": ["s1", "s2", "s3", "s4"]},
                    "labels": {"type": "array", "items": {"type": "string"}},
                    "assignee": {"type": "string"},
                },
                "required": ["project", "summary", "severity"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scale_deployment",
            "description": "Change the replica count of a deployment.",
            "parameters": {
                "type": "object",
                "properties": {
                    "namespace": {"type": "string"},
                    "deployment": {"type": "string"},
                    "replicas": {"type": "integer"},
                    "wait_for_ready": {"type": "boolean"},
                },
                "required": ["namespace", "deployment", "replicas"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "book_hotel",
            "description": "Reserve a hotel room.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "check_in": {"type": "string"},
                    "nights": {"type": "integer"},
                    "guests": {"type": "integer"},
                    "max_price_per_night": {"type": "number"},
                },
                "required": ["city", "check_in", "nights", "guests"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "translate_text",
            "description": "Translate text into a target language.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "target_language": {"type": "string"},
                    "formality": {
                        "type": "string",
                        "enum": ["default", "formal", "informal"],
                    },
                },
                "required": ["text", "target_language"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_transactions",
            "description": "List account transactions in a date range.",
            "parameters": {
                "type": "object",
                "properties": {
                    "account_id": {"type": "string"},
                    "start_date": {"type": "string"},
                    "end_date": {"type": "string"},
                    "min_amount": {"type": "number"},
                    "include_pending": {"type": "boolean"},
                },
                "required": ["account_id", "start_date", "end_date"],
            },
        },
    },
]

# (user request, name of the tool the request should elicit). The expected tool
# name is carried through into the frozen prompt set so that the later quality
# pass can score call fidelity without rebuilding anything.
TOOL_QUERIES = [
    ("Find me a nonstop flight from Frankfurt to London on 2026-03-14, refundable only.", "search_flights"),
    ("I need to get from Munich to Lisbon on 2026-04-02, at most one layover.", "search_flights"),
    ("Book a flight from Berlin to Zurich for 2026-05-20.", "search_flights"),
    ("Put a 45 minute design review on my calendar for 2026-03-10 at 14:00 with priya and sam.", "create_calendar_event"),
    ("Schedule a private one hour retrospective on 2026-03-18 starting at 09:30.", "create_calendar_event"),
    ("Add a 30 minute standup to my calendar on 2026-03-09 at 10:00.", "create_calendar_event"),
    ("What was the p99 request latency over the last 30 minutes for the checkout service?", "query_metrics"),
    ("Show me average CPU utilization for the last 120 minutes.", "query_metrics"),
    ("Give me the p95 queue depth over the past 15 minutes.", "query_metrics"),
    ("How much is 2500 euros in Japanese yen?", "convert_currency"),
    ("Convert 149.99 US dollars to Swiss francs.", "convert_currency"),
    ("Email the platform team at platform@example.com about the failed deploy, high priority.", "send_email"),
    ("Send a note to alex@example.com with the subject Weekly update and a short status body.", "send_email"),
    ("What is the weather in Amsterdam three days from now, in metric units?", "get_weather_forecast"),
    ("Give me the forecast for Oslo one day ahead.", "get_weather_forecast"),
    ("Run a query counting orders per region from the orders table, limit 100.", "run_sql_query"),
    ("Query the warehouse for the ten largest invoices in the last quarter.", "run_sql_query"),
    ("Open an s2 ticket in the INFRA project about intermittent 502s on the gateway.", "create_ticket"),
    ("File an s1 ticket for the payments outage and assign it to the on-call engineer.", "create_ticket"),
    ("Scale the api deployment in the production namespace to 12 replicas and wait for readiness.", "scale_deployment"),
    ("Bring the worker deployment in staging down to 2 replicas.", "scale_deployment"),
    ("Reserve a hotel in Barcelona for two guests checking in 2026-06-11 for three nights under 200 per night.", "book_hotel"),
    ("Find me a room in Vienna for one guest, check in 2026-07-01, five nights.", "book_hotel"),
    ("Translate the sentence 'The deployment has been rolled back' into German, formal.", "translate_text"),
    ("Translate 'Please confirm receipt' into Japanese.", "translate_text"),
    ("List transactions on account ACC-8842 between 2026-01-01 and 2026-01-31 over 500 euros.", "list_transactions"),
    ("Show pending and settled transactions for ACC-1109 from 2026-02-01 to 2026-02-28.", "list_transactions"),
]

TOOL_SYSTEM_PROMPT = (
    "You are a precise assistant with access to tools. When a user request can be "
    "satisfied by a tool, call it with correctly typed arguments. Do not invent "
    "parameters that are not in the schema, and do not ask for confirmation."
)
