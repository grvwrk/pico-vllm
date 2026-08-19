# pico-vllm — Architecture

```mermaid
flowchart TB
    subgraph Client
        A[Incoming request]
    end

    subgraph Server["server/"]
        B[FastAPI api.py]
        C[Async request queue]
    end

    subgraph Engine["engine/"]
        D[LLMEngine step loop]
    end

    subgraph Scheduler["scheduler/"]
        E[Scheduler admit / evict]
        F[Batch builder]
    end

    subgraph Core["core/"]
        G[BlockManager free-list + block table]
        H[KV Cache physical block storage]
        I[Paged Attention gather + SDPA]
        J[Model Wrapper]
    end

    subgraph Benchmarks["benchmarks/"]
        K[Workload generator]
        L[Naive vs continuous vs vLLM]
        M[Metrics: throughput, latency, fragmentation]
    end

    A --> B --> C --> D
    D --> E
    E --> F
    E -->|allocate/free blocks| G
    G --> H
    F --> J
    J --> I
    I -->|reads via block table| H
    D -->|generated tokens| B

    K --> L --> D
    L --> M

    style Core fill:#f1efe8,stroke:#444441
    style Scheduler fill:#f1efe8,stroke:#5f5e5a
    style Engine fill:#f1efe8,stroke:#5f5e5a
    style Server fill:#f1efe8,stroke:#888780
    style Benchmarks fill:#f1efe8,stroke:#888780
```

## Flow summary

A request comes in through the server, the engine's step loop asks the
scheduler to decide which sequences run this iteration, the scheduler talks
to the block manager to allocate/free KV-cache blocks, the model wrapper
runs a forward pass using paged attention (gathering blocks from the KV
cache), and generated tokens flow back out. The benchmarks suite sits
outside this loop, driving synthetic workloads through the same engine to
produce comparison numbers against naive batching and real vLLM.
