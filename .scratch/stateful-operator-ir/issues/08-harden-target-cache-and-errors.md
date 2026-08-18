# 08 — Harden Target, cache, and error contracts

**What to build:** Complete deterministic Target resolution, structured diagnostics, semantic/binary/tuning identity separation, concurrent publication, and failure cleanup.

**Blocked by:** 05 — Add packed variable-length execution.

**Status:** ready-for-agent

- [ ] Optional Target resolves only from bound tensors when omitted and rejects mismatched explicit constraints.
- [ ] H20 resolution checks approved product family and CUDA capability; H100, unknown products, CPU, ROCm, and unidentifiable MIG fail stably.
- [ ] Error codes, canonical public-IR paths, and JSON-safe details follow deterministic validation order.
- [ ] Semantic identity includes schema, canonical forward/backward, concrete metadata, sequence representation, active gradients, final-State mode, Target architecture, and relevant options.
- [ ] Runtime offset values, device index, object identity, source paths, and messages are excluded.
- [ ] Binary identity includes compiler, TileLang, CUDA ABI, and compatibility fingerprint.
- [ ] Tuning identity is isolated by exact product variant, resources, software stack, and clock policy.
- [ ] Shared artifacts use process-safe locking and atomic publication.
- [ ] Transient compile/driver/resource failures remove temporary files and never create negative cache entries.
- [ ] CPU/concurrency tests cover deterministic identity, schema invalidation, publication, cleanup, and structured errors.
