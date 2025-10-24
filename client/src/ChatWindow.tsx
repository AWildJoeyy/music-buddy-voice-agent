import React, { useCallback, useEffect, useRef, useState } from "react";
import {
  Mic, MicOff, Send, Square, Settings2, Bot, User, ClipboardCopy, Check,
  Sparkles, RefreshCw, Upload, Loader2, Play, Pause, SkipBack, SkipForward,
  Volume2, StopCircle
} from "lucide-react";

declare global {
  interface Window {
    e2e: {
      send: (payload: any) => Promise<any>;
      onPython: (handler: (msg: any) => void) => () => void;
      pickAudio: () => Promise<string[]>;
      tts: (id: string, text: string) => Promise<any>;
      asrStart: (fmt?: "webm" | "wav") => Promise<any>;
      asrChunk: (data_b64: string) => Promise<any>;
      asrEnd: () => Promise<any>;
    };
  }
}

type ChatMessage = { id: string; role: "user" | "assistant" | "system"; content: string };
const uid = () => Math.random().toString(36).slice(2);
const USER_NAME = "Joey";

const Button = ({
  className = "",
  children,
  ...props
}: React.ButtonHTMLAttributes<HTMLButtonElement> & { className?: string }) => (
  <button
    {...props}
    className={`inline-flex items-center gap-2 rounded-2xl px-3 py-2 border border-zinc-700/50 bg-zinc-800 hover:bg-zinc-700 disabled:opacity-50 ${className}`}
  >
    {children}
  </button>
);

const Textarea = React.forwardRef<HTMLTextAreaElement, React.TextAreaHTMLAttributes<HTMLTextAreaElement>>(
  ({ className = "", style, ...props }, ref) => (
    <textarea
      ref={ref}
      rows={1}
      {...props}
      style={{ ...style, scrollbarWidth: "none" }}
      className={`w-full resize-none rounded-2xl bg-zinc-900/60 border border-zinc-700/50 px-3 py-2 outline-none focus:ring-2 focus:ring-indigo-500 overflow-hidden ${className}`}
    />
  )
);
Textarea.displayName = "Textarea";

const fmt = (ms?: number) => {
  const n = Number.isFinite(ms) && (ms as number) > 0 ? Math.floor((ms as number)/1000) : 0;
  const m = Math.floor(n/60).toString();
  const s = (n%60).toString().padStart(2,"0");
  return `${m}:${s}`;
};

// sanitize playlist names
const cleanPlaylist = (raw: string) =>
  raw.replace(/^[\s'"]+|[\s'"]+$/g, "")
     .replace(/^(my|the)\s+/i, "")
     .replace(/\s+/g, " ")
     .trim();

export default function ChatWindow() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);

  const [vol, setVol] = useState(80);
  const [posMs, setPosMs] = useState(0);
  const [durMs, setDurMs] = useState(0);

  const greetedRef = useRef(false);
  const taRef = useRef<HTMLTextAreaElement | null>(null);
  const endRef = useRef<HTMLDivElement | null>(null);

  // TTS + action gating
  const ttsActiveRef = useRef(false);
  const actionQueueRef = useRef<any[]>([]);
  const spokenSetRef = useRef<Set<string>>(new Set());

  // play de-dupe
  const playSuppressedRef = useRef(false);
  const playSuppressionTimerRef = useRef<number | null>(null);

  const playPendingRef = useRef(false);
  const deferredSpeechRef = useRef<{ id: string; text: string } | null>(null);

  const nowPlayingRef = useRef<string | null>(null);
  const lastAssistantRef = useRef<{ id: string; content: string } | null>(null);
  const playLockRef = useRef(false);

  // --- one-shot PTT state ---
  const [recording, setRecording] = useState(false);
  const [asrUsed, setAsrUsed] = useState(false); // disable PTT after first transcript
  const mediaRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  useEffect(() => {
    const ta = taRef.current; if (!ta) return;
    ta.style.height = "0px";
    ta.style.height = Math.min(200, ta.scrollHeight) + "px";
  }, [input]);

  const scrollToEnd = () => endRef.current?.scrollIntoView({ behavior: "smooth" });

  const sendAction = useCallback((payload: any) => {
    if (ttsActiveRef.current) actionQueueRef.current.push(payload);
    else window.e2e.send(payload);
  }, []);

  const flushActionQueue = useCallback(() => {
    if (ttsActiveRef.current) return;
    const q = actionQueueRef.current;
    while (q.length) window.e2e.send(q.shift());
  }, []);

  const speakBlocking = useCallback(async (id: string, text: string) => {
    if (!text?.trim()) return;
    if (playPendingRef.current) { deferredSpeechRef.current = { id, text }; return; }
    ttsActiveRef.current = true;
    try { await window.e2e.tts(id, text); }
    finally { ttsActiveRef.current = false; flushActionQueue(); }
  }, [flushActionQueue]);

  const speakOnce = useCallback(async (text: string) => {
    const key = (text || "").trim();
    if (!key || spokenSetRef.current.has(key)) return;
    spokenSetRef.current.add(key);
    await speakBlocking(uid(), key);
  }, [speakBlocking]);

  // central play trigger
  const playPlaylist = (nameRaw: string) => {
    const name = cleanPlaylist(nameRaw);
    if (!name) return;
    if (playLockRef.current) return;
    playLockRef.current = true;

    playSuppressedRef.current = true;
    if (playSuppressionTimerRef.current) window.clearTimeout(playSuppressionTimerRef.current);
    playSuppressionTimerRef.current = window.setTimeout(() => { playSuppressedRef.current = false; }, 4000);

    playPendingRef.current = true;
    sendAction({ type: "music_play_playlist", playlist: name });
    window.setTimeout(() => { playLockRef.current = false; }, 1000);
  };

  // tool extraction from assistant text
  const runToolFromText = (text: string): boolean => {
    if (playSuppressedRef.current) return false;
    const m = text.match(/play_playlist\(['"]([\w\- ]+)['"]\)/i);
    const m2 = text.match(/play (?:from (?:my|the) )?([\w\- ]+)\s+playlist/i);
    const m3 = text.match(/\bplay\s+([\w\- ]+)\b/i);
    const pickRaw = (m?.[1] || m2?.[1] || m3?.[1] || "").trim();
    const pick = cleanPlaylist(pickRaw);
    if (pick) { playPlaylist(pick); return true; }
    if (/pause\b/i.test(text))  { sendAction({ type: "music_ctrl", cmd: "pause" }); return true; }
    if (/\bresume\b|\bcontinue\b/i.test(text)) { sendAction({ type: "music_ctrl", cmd: "resume" }); return true; }
    if (/\bnext\b/i.test(text)) { sendAction({ type: "music_ctrl", cmd: "next" }); return true; }
    if (/\bprev(?:ious)?\b/i.test(text)) { sendAction({ type: "music_ctrl", cmd: "prev" }); return true; }
    return false;
  };

  // IPC handler
  const handlePythonMessage = useCallback((msg: any) => {
    const t = msg?.type as string; if (!t) return;

    if (t === "stderr") return;

    if ((t === "ready" || t === "diag") && !greetedRef.current) {
      greetedRef.current = true;
      const id = uid();
      const content = `Hello ${USER_NAME}! How can I help with your music today?`;
      setMessages((m) => [...m, { id, role: "assistant", content }]);
      speakBlocking(id, content);
      return;
    }

    if (t === "chat_chunk") {
      const id = msg.id, piece = msg.text ?? ""; if (!piece) return;
      setMessages((m) => {
        const idx = m.findIndex((x) => x.role === "assistant" && x.id === id);
        if (idx === -1) { lastAssistantRef.current = { id, content: piece }; return [...m, { id, role: "assistant", content: piece }]; }
        const copy = m.slice(); const merged = copy[idx].content + piece; copy[idx] = { ...copy[idx], content: merged };
        lastAssistantRef.current = { id, content: merged }; return copy;
      });
      scrollToEnd();
      return;
    }

    if (t === "chat_end") {
      setStreaming(false);
      const la = lastAssistantRef.current;
      if (!la || !la.content.trim()) return;
      const triggeredPlay = runToolFromText(la.content);
      if (triggeredPlay) { deferredSpeechRef.current = { id: la.id, text: la.content }; }
      else { speakBlocking(la.id, la.content); }
      return;
    }

    if (t === "music_player") {
      const r = msg.result || {};
      setPosMs(0); setDurMs(0); // reset bar on fresh play
      if (Number.isFinite(r.volume)) setVol(r.volume);
      if (Number.isFinite(r.position_ms)) setPosMs(r.position_ms);
      if (Number.isFinite(r.duration_ms)) setDurMs(r.duration_ms);
      if (typeof r.current === "string") nowPlayingRef.current = r.current;

      const line =
        r.status === "playing" ? `Playing ${r.count ?? "some"} track(s) from playlists/${r.playlist ?? "?"}.` :
        r.status === "empty" ? `No tracks found in playlists/${r.playlist ?? "?"}.` :
        r.status === "unavailable" ? `Music player is not available.` :
        `Player: ${r.status || "unknown"}`;

      const id = uid();
      setMessages((m) => [...m, { id, role: "assistant", content: line }]);

      playSuppressedRef.current = false;
      if (playSuppressionTimerRef.current) { window.clearTimeout(playSuppressionTimerRef.current); playSuppressionTimerRef.current = null; }

      if (playPendingRef.current) {
        playPendingRef.current = false;
        const pending = deferredSpeechRef.current; deferredSpeechRef.current = null;
        (async () => {
          if (pending) await speakBlocking(pending.id, pending.text);
          await speakBlocking(id, line);
        })();
      } else {
        speakBlocking(id, line);
      }
      return;
    }

    if (t === "music_now_playing") {
      const name = String(msg.name ?? "").trim();
      if (name) {
        setPosMs(0); setDurMs(0); // reset bar on track change
        nowPlayingRef.current = msg.current || name;
        const line = `Now playing: ${name}`;
        const id = uid();
        setMessages((m) => [...m, { id, role: "assistant", content: line }]);

        playSuppressedRef.current = false;
        if (playSuppressionTimerRef.current) { window.clearTimeout(playSuppressionTimerRef.current); playSuppressionTimerRef.current = null; }

        if (playPendingRef.current) {
          playPendingRef.current = false;
          const pending = deferredSpeechRef.current; deferredSpeechRef.current = null;
          (async () => {
            if (pending) await speakBlocking(pending.id, pending.text);
            await speakBlocking(id, line);
          })();
        } else {
          speakBlocking(id, line);
        }
      }
      return;
    }

    if (t === "music_status") {
      const s = msg.status || {};
      if (Number.isFinite(s.volume)) setVol(s.volume);
      if (Number.isFinite(s.position_ms)) setPosMs(s.position_ms);
      if (Number.isFinite(s.duration_ms)) setDurMs(s.duration_ms);
      if (typeof s.current === "string" && s.current !== nowPlayingRef.current) {
        setPosMs(0); setDurMs(0); // reset when current changes
        nowPlayingRef.current = s.current;
        const line = `Now playing: ${s.current.split(/[\\/]/).pop()}`;
        const id = uid();
        setMessages((m) => [...m, { id, role: "assistant", content: line }]);
        if (playPendingRef.current) {
          playPendingRef.current = false;
          const pending = deferredSpeechRef.current; deferredSpeechRef.current = null;
          (async () => {
            if (pending) await speakBlocking(pending.id, pending.text);
            await speakBlocking(id, line);
          })();
        } else {
          speakBlocking(id, line);
        }
      }
      return;
    }

    if (t === "asr_final") {
      const text = (msg.text ?? "").trim();
      if (text) {
        // show it in the transcript
        setMessages((m) => [...m, { id: uid(), role: "user", content: text }]);

        const handled = tryDirectUserIntent(text);
        if (!handled) {
          const id = uid();
          setStreaming(true);
          window.e2e.send({ type: "chat", id, messages: [{ role: "user", content: text }] });
        }
      }
      setAsrUsed(true);     
      return;
    }
    if (t === "asr_disabled") { setAsrUsed(true); return; }
  }, [sendAction, speakBlocking]);

  useEffect(() => {
    const off = window.e2e.onPython(handlePythonMessage);
    window.e2e.send({ type: "diag" });
    return () => off();
  }, [handlePythonMessage]);

  // direct user intent (skips LLM)
  const tryDirectUserIntent = (userText: string): boolean => {
    const text = userText.toLowerCase().trim();
    const m = text.match(/play (?:from (?:my|the) )?([\w\- ]+)\s+playlist/);
    const m2 = text.match(/^play\s+([\w\- ]+)$/);
    const clean = (s: string) => s.replace(/^(my|the)\s+/i, "").trim();

    const pick = clean((m?.[1] || m2?.[1] || "").trim());
    if (pick) { playPlaylist(pick); return true; }

    if (/^pause\b/.test(text))             { sendAction({ type: "music_ctrl", cmd: "pause" });  return true; }
    if (/^(resume|continue)\b/.test(text)) { sendAction({ type: "music_ctrl", cmd: "resume" }); return true; }
    if (/^next\b/.test(text))              { sendAction({ type: "music_ctrl", cmd: "next" });   return true; }
    if (/^(prev|previous)\b/.test(text))   { sendAction({ type: "music_ctrl", cmd: "prev" });   return true; }
    return false;
  };

  const sendChat = (userText: string) => {
    const handledDirect = tryDirectUserIntent(userText);
    setMessages((m) => [...m, { id: uid(), role: "user", content: userText }]);
    if (handledDirect) return;
    const id = uid();
    setStreaming(true);
    window.e2e.send({ type: "chat", id, messages: [{ role: "user", content: userText }] });
  };

  const onUploadMp3 = async () => {
    const paths = (await window.e2e.pickAudio()) || [];
    if (!paths.length) return;
    setMessages((m) => [...m, { id: uid(), role: "system", content: `Uploading ${paths.length} file(s)…` }]);
    window.e2e.send({ type: "music_auto_bucket", paths });
  };

  // --- one-shot PTT handlers ---
  const startOneShotRecording = async () => {
    if (recording || asrUsed) return;
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    streamRef.current = stream;
    await window.e2e.asrStart("webm");
    const mr = new MediaRecorder(stream, { mimeType: "audio/webm" });
    mediaRef.current = mr;

    mr.ondataavailable = async (e) => {
      if (!e.data || e.data.size === 0) return;
      const ab = await e.data.arrayBuffer();
      const b64 = arrayBufferToBase64(ab);
      await window.e2e.asrChunk(b64);
    };
    mr.onstop = async () => {
      await window.e2e.asrEnd();
      stream.getTracks().forEach(t => t.stop());
      streamRef.current = null;
      setRecording(false);
    };

    mr.start(250);
    setRecording(true);
  };

  const stopOneShotRecording = async () => {
    if (!recording) return;
    mediaRef.current?.stop();
  };

  const togglePTT = async () => {
    if (asrUsed) return;          // one shot only
    if (!recording) await startOneShotRecording();
    else await stopOneShotRecording();
  };

  const canSend = input.trim().length > 0 && !streaming;

  const safeDur = Number.isFinite(durMs) && durMs > 0 ? durMs : 0;
  const safePos = Math.max(0, Math.min(posMs || 0, safeDur));

  return (
    <div className="h-screen w-full bg-zinc-950 text-zinc-100 flex flex-col">
      <div className="flex items-center justify-between px-4 py-3 border-b border-zinc-800">
        <div className="flex items-center gap-3">
          <div className="h-8 w-8 rounded-xl bg-indigo-600 grid place-items-center shadow">
            <Sparkles className="h-4 w-4" />
          </div>
        </div>
        <div className="flex items-center gap-2">
          <Button onClick={onUploadMp3}><Upload className="h-4 w-4" /> Upload MP3</Button>
          <Button className="hidden sm:flex"><Settings2 className="h-4 w-4" /> Settings</Button>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-3">
        {messages.map((m) => (
          <div key={m.id} className={`max-w-3xl mx-auto flex items-start gap-3 ${m.role === "user" ? "flex-row-reverse" : ""}`}>
            <div className={`rounded-xl p-2 ${m.role === "user" ? "bg-indigo-600" : m.role === "assistant" ? "bg-zinc-800" : "bg-amber-900/40"}`}>
              {m.role === "user" ? <User className="h-4 w-4" /> : m.role === "assistant" ? <Bot className="h-4 w-4" /> : <Sparkles className="h-4 w-4" />}
            </div>
            <div className={`rounded-2xl px-4 py-3 border border-zinc-800/60 bg-zinc-900/50 w-full ${m.role === "user" ? "text-right" : "text-left"}`}>
              <div className="whitespace-pre-wrap leading-relaxed">{m.content}</div>
              {m.role === "assistant" && (
                <div className="mt-2 flex gap-2 text-xs text-zinc-400">
                  <MsgCopy text={m.content} />
                  <button className="inline-flex items-center gap-1 hover:text-zinc-200" onClick={() => speakBlocking(m.id, m.content)}>
                    <Play className="h-3 w-3" /> Speak
                  </button>
                </div>
              )}
            </div>
          </div>
        ))}
        <div ref={endRef} />
      </div>

      <div className="border-t border-zinc-800 p-3 space-y-3">
        <div className="max-w-5xl mx-auto flex items-end gap-2 flex-wrap">
          <Button onClick={togglePTT} disabled={asrUsed} className={recording ? "bg-red-800" : ""}>
            {recording ? <MicOff className="h-4 w-4" /> : <Mic className="h-4 w-4" />}
            {asrUsed ? "Push-to-talk (F9)" : recording ? "Recording… (click to stop)" : "Push-to-talk (F9)"}
          </Button>

          <div className="flex-1 min-w-[260px]">
            <Textarea
              ref={taRef}
              placeholder={"Message the bot"}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey && canSend) {
                  e.preventDefault();
                  const text = input.trim();
                  setInput("");
                  sendChat(text);
                }
              }}
            />
          </div>

          <Button onClick={() => { if (!canSend) return; const text = input.trim(); setInput(""); sendChat(text); }} disabled={!canSend}>
            <Send className="h-4 w-4" /> Send
          </Button>

          {streaming && (
            <Button onClick={() => window.e2e.send({ type: "stop", id: "chat" })} className="bg-red-800 hover:bg-red-700">
              <Square className="h-4 w-4" /> Stop
            </Button>
          )}

          <Button
            onClick={() =>
              window.e2e.send({
                type: "chat",
                id: uid(),
                messages: messages.filter((m) => m.role !== "assistant").map((m) => ({ role: m.role, content: m.content })),
              })
            }
          >
            <RefreshCw className="h-4 w-4" /> Retry
          </Button>
        </div>

        <div className="max-w-5xl mx-auto flex items-center gap-3">
          <div className="flex items-center gap-2">
            <Button onClick={() => sendAction({ type: "music_ctrl", cmd: "prev" })}><SkipBack className="h-4 w-4" /></Button>
            <Button onClick={() => sendAction({ type: "music_ctrl", cmd: "pause" })}><Pause className="h-4 w-4" /></Button>
            <Button onClick={() => sendAction({ type: "music_ctrl", cmd: "resume" })}><Play className="h-4 w-4" /></Button>
            <Button onClick={() => sendAction({ type: "music_ctrl", cmd: "next" })}><SkipForward className="h-4 w-4" /></Button>
            <Button onClick={() => sendAction({ type: "music_ctrl", cmd: "stop" })}><StopCircle className="h-4 w-4" /></Button>
          </div>

          <div className="flex items-center gap-3 flex-1 min-w-[240px]">
            <span className="text-xs text-zinc-400 w-10 text-right">{fmt(safePos)}</span>
            <input
              type="range"
              min={0}
              max={safeDur || 1}
              value={Math.min(safePos, safeDur || 0)}
              onChange={(e) => setPosMs(Number(e.target.value))}
              onPointerUp={(e) => {
                const v = Math.min(Number((e.target as HTMLInputElement).value), safeDur || 0);
                sendAction({ type: "music_ctrl", cmd: `seek:${v}` });
              }}
              className="flex-1 accent-indigo-500"
            />
            <span className="text-xs text-zinc-400 w-10">{fmt(safeDur)}</span>
          </div>

          <div className="flex items-center gap-2">
            <Button onClick={() => sendAction({ type: "music_ctrl", cmd: `vol:${vol}` })}>
              <Volume2 className="h-4 w-4" /> {vol}
            </Button>
            <input
              type="range"
              min={0}
              max={100}
              value={vol}
              onChange={(e) => setVol(Number(e.target.value))}
              onMouseUp={() => sendAction({ type: "music_ctrl", cmd: `vol:${vol}` })}
              onTouchEnd={() => sendAction({ type: "music_ctrl", cmd: `vol:${vol}` })}
              className="w-32 accent-indigo-500"
            />
          </div>
        </div>
      </div>
    </div>
  );
}

function MsgCopy({ text }: { text: string }) {
  const [busy, setBusy] = useState(false);
  const [copied, setCopied] = useState(false);
  return (
    <button
      className="inline-flex items-center gap-1 hover:text-zinc-200"
      onClick={async () => {
        try { setBusy(true); await navigator.clipboard.writeText(text); setCopied(true); setTimeout(() => setCopied(false), 1200); }
        finally { setBusy(false); }
      }}
      disabled={busy}
      title="Copy"
    >
      {copied ? <Check className="h-3 w-3" /> : busy ? <Loader2 className="h-3 w-3 animate-spin" /> : <ClipboardCopy className="h-3 w-3" />} Copy
    </button>
  );
}

function arrayBufferToBase64(buffer: ArrayBuffer): string {
  const bytes = new Uint8Array(buffer); let binary = ""; const chunk = 0x8000;
  for (let i=0; i<bytes.length; i+=chunk) binary += String.fromCharCode.apply(null, Array.from(bytes.subarray(i, i+chunk)) as any);
  return btoa(binary);
}
