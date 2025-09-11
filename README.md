# Coding Buddy 

Local, voice-aware coding buddy that listens for developer intents, proposes minimal code patches, and explains the “why” in a few concise bullets. This document outlines the **planned architecture, libraries, and models**.

---

## Vision & Core Capabilities
- **Hands-free prompts:** Voice in; natural language requests.
- **Debug-first workflow:** Fault localization → minimal **unified diff** → short rationale.
- **Code reasoning:** Context-aware edits that reference the current file and relevant neighbors.
- **Private by default:** Fully offline mode; no data leaves the machine.

---

## Planned System Architecture
1. **Audio Frontend**
   - **VAD:** Gate mic with Voice Activity Detection to segment speech.
   - **ASR:** Transcribe voiced chunks; return partials for responsiveness.

2. **Intent Router**
   - Classify utterances into: *generate*, *debug/fix*, *explain*, *test*, *search file*.
   - Route to appropriate prompting template.

3. **Context Builder**
   - Gather current file slice, short repository file map, recent error/failing test output.
   - Apply hard caps (tokens + files) and redact secrets.

4. **Coder Engine**
   - Balanced code LLM generates **unified diff** (preferred) or patch block.
   - Four–six bullet **“why this works”** explanation.

6. **Response Layer**
   - Stream diff + rationale to client UI.
   - Offer **Apply / Stage / Discard** actions.

---

## Planned Libraries & Components

### Backend & APIs
- **FastAPI** (HTTP + WebSocket streaming)
- **Pydantic** (schemas), **Uvicorn** (server), **Starlette** (WS)

### Audio
- **VAD:** *Silero VAD* (CPU)
- **ASR:** *faster-whisper* with **distil-large-v3** 

### Code LLM 
- **Qwen2.5-Coder-7B-Instruct (4-bit)**  

### Repo Context & Analysis
- **ripgrep** ,**tree-sitter** 
- **Path allow-list** + size caps for safe indexing
- **Secret scrubbing** (regex patterns for tokens/keys)

---

## Safety & Privacy 
- **Offline mode**: no external calls when enabled.
- **Filesystem guardrails**: write within workspace only; deny shell/network ops from code prompts.
- **Secret redaction**: hiding credentials, API codes, and any personal info.
- **Minimal diff policy**: no sweeping refactors; edits limited to target files unless explicitly allowed.

---

## Performance Envelope 
- **GPU VRAM** ~8–11 GB during generation (LLM + KV + ASR + overhead)
- **Throughput** ~20–35 tok/s (7B @ 4-bit, loader-dependent)
- **CPU** low–moderate (ASR I/O, VAD, server; spikes if tests run)
- **Best for**: single active user; short, surgical edits