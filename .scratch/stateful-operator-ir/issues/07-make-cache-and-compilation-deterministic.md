# 07 — Make cache and compilation deterministic

**What to build:** Give compositional generated programs stable identity, deterministic source, safe concurrent publication, separate tuning records, and correct failure cleanup.

**Blocked by:** 04 — Lower derived backward and state continuation.

**Status:** ready-for-agent

- [ ] Cache identity includes compiler schema version, canonical forward IR, derived backward identity, execution class, concrete metadata, active gradient mode, Target architecture, final-state mode, and relevant options.
- [ ] Frontend identity, device index, diagnostics, source paths, and formatting are excluded.
- [ ] Identical identity inputs generate byte-identical semantic source across processes.
- [ ] Autotuning results are versioned auxiliary artifacts keyed separately from semantic identity.
- [ ] Shared artifacts use process-safe locking and atomic publication.
- [ ] Concurrent callers never observe partial or corrupt artifacts.
- [ ] Transient compilation, driver, and resource failures remove temporary artifacts and never poison the cache.
- [ ] Deterministic analysis failures remain ProgramAnalysis data rather than cache entries.
- [ ] CPU tests exercise identity changes, process concurrency, atomicity, schema invalidation, and cleanup.
