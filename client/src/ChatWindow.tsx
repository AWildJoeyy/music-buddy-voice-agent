import React, { useEffect, useMemo, useRef, useState } from "react";
import { motion } from "framer-motion";
import {
  Mic, MicOff, Send, Square, Play, Settings2, Bot, User,
  ClipboardCopy, Check, Sparkles, RefreshCw, Code, Loader2
} from "lucide-react";

declare global {
  interface Window {
    e2e: {
      chat: (id: string, messages: {role:"user"|"assistant"|"system"; content:string}[], stream?: boolean) => Promise<any>;
      stop: (id: string) => Promise<any>;
      analyze: (id: string, paths: string[]) => Promise<any>;
      tts: (id: string, text: string) => Promise<any>;
      pickFiles: () => Promise<string[]>;
      asrStart: (fmt?: "webm"|"wav") => Promise<any>;
      asrChunk: (data_b64: string) => Promise<any>;
      asrEnd: () => Promise<any>;
      onPython: (handler: (msg:any)=>void) => () => void;
      onHotkey: (handler: (ev:any)=>void) => () => void;
    }
  }
}

type ChatMessage = {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  audioUrl?: string;
};

const QUIET_PY_STDERR = true; // hide backend warnings in UI
const uid = () => Math.random().toString(36).slice(2);

const Button = ({
  className = "", disabled, onClick, children, title, type
}: React.ButtonHTMLAttributes<HTMLButtonElement> & { className?: string }) => (
  <button
    type={type}
    title={title}
    onClick={onClick}
    disabled={disabled}
    className={`inline-flex items-center gap-2 rounded-2xl px-3 py-2 shadow-sm border border-zinc-700/50 bg-zinc-800 hover:bg-zinc-700 disabled:opacity-50 ${className}`}
  >
    {children}
  </button>
);

const Textarea = React.forwardRef<HTMLTextAreaElement, React.TextareaHTMLAttributes<HTMLTextAreaElement>>(
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

export default function ChatWindow() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [copyId, setCopyId] = useState<string | null>(null);
  const [ttsBusy, setTtsBusy] = useState<string | null>(null);
  const [recording, setRecording] = useState(false);

  const taRef = useRef<HTMLTextAreaElement | null>(null);
  const endRef = useRef<HTMLDivElement | null>(null);
  const mediaRef = useRef<MediaRecorder | null>(null);

  // Auto-grow & hide scrollbar
  useEffect(() => {
    const ta = taRef.current;
    if (!ta) return;
    ta.style.height = "0px";
    ta.style.height = Math.min(200, ta.scrollHeight) + "px";
  }, [input]);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages.length, streaming]);

  // Listen to Python worker
  useEffect(() => {
    const off = window.e2e.onPython((msg) => {
      if (msg.type === "ready") {
        setMessages((m)=>[...m,{id:uid(), role:"system", content:"Python worker ready"}]);
      } else if (msg.type === "diag") {
        const s = msg.status || {};
        setMessages((m)=>[...m,{id:uid(), role:"system", content:`Modules → agent:${s.agent} analyzer:${s.analyzer} tts:${s.tts} asr:${s.asr} (cwd: ${s.cwd})`}]);
      } else if (msg.type === "chat_chunk") {
        const id = msg.id as string;
        setMessages((m) => {
          const last = m[m.length-1];
          if (!last || last.id !== id) {
            return [...m, { id, role:"assistant", content: msg.text ?? "" }];
          } else {
            return m.map(x => x.id === id ? { ...x, content: x.content + (msg.text ?? "") } : x);
          }
        });
      } else if (msg.type === "chat_end") {
        setStreaming(false);
        // Auto-speak the latest assistant message
        const last = [...messages].reverse().find(m => m.role === "assistant");
        if (last && last.content.trim()) {
          window.e2e.tts(last.id, last.content);
        }
      } else if (msg.type === "analyze_result") {
        setMessages((m)=>[...m,{id:uid(), role:"assistant", content: msg.summary ?? "(no summary)"}]);
      } else if (msg.type === "tts_result") {
        const mime = msg.mime || "audio/wav";
        const dataUrl = `data:${mime};base64,${msg.audio_b64}`;
        setMessages((prev) => {
          const i = [...prev].reverse().findIndex(p => p.role === "assistant");
          if (i === -1) return [...prev, { id: uid(), role:"system", content:`(voice ready)` }];
          const idx = prev.length - 1 - i;
          const copy = prev.slice();
          copy[idx] = { ...copy[idx], audioUrl: dataUrl };
          return copy;
        });
        try { new Audio(dataUrl).play(); } catch {}
      } else if (msg.type === "asr_final") {
        const text = (msg.text || "").trim();
        if (text) {
          const userMsg: ChatMessage = { id: uid(), role: "user", content: text };
          setMessages((m)=>[...m, userMsg]);
          chatCompletion([...messages, userMsg]);
        }
        setRecording(false);
      } else if (msg.type === "stderr") {
        if (QUIET_PY_STDERR) return; // suppress noisy backend warnings
        setMessages((m)=>[...m,{id:uid(), role:"system", content:`[py] ${msg.data}`}]);
      }
    });
    return () => off();
  }, [messages]);

  // Hotkey (F9) toggle
  useEffect(() => {
    const off = window.e2e.onHotkey((ev) => {
      if (ev?.type === "ptt-toggle") {
        if (recording) stopRecording();
        else startRecording();
      }
    });
    return () => off();
  }, [recording]);

  const sendUserMessage = async (text: string) => {
    if (!text.trim()) return;
    const userMsg: ChatMessage = { id: uid(), role: "user", content: text };
    setMessages((m) => [...m, userMsg]);
    setInput("");
    await chatCompletion([...messages, userMsg]);
  };

  const chatCompletion = async (history: ChatMessage[]) => {
    try {
      setStreaming(true);
      const id = uid();
      await window.e2e.chat(id, history.map(({role,content})=>({role,content})), true);
    } catch (e) {
      console.error(e);
      setMessages((m)=>[...m,{id:uid(),role:"system",content:"⚠️ Chat error."}]);
      setStreaming(false);
    }
  };

  const stop = async () => {
    try { setStreaming(false); await window.e2e.stop("chat"); } catch {}
  };

  const speak = async (m: ChatMessage) => {
    try {
      setTtsBusy(m.id);
      await window.e2e.tts(m.id, m.content);
    } finally {
      setTtsBusy(null);
    }
  };

  const onAnalyzeCode = async () => {
    const paths = await window.e2e.pickFiles();
    if (!paths || paths.length === 0) return;
    setMessages((m)=>[...m,{id:uid(),role:"system",content:`Analyzing ${paths.length} file(s)…`}]);
    await window.e2e.analyze(uid(), paths);
  };

  // --- Push-to-talk helpers ---
  const startRecording = async () => {
    if (recording) return;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      await window.e2e.asrStart("webm");
      const mr = (mediaRef.current = new MediaRecorder(stream, { mimeType: "audio/webm" }));
      mr.ondataavailable = async (e) => {
        if (!e.data || e.data.size === 0) return;
        const ab = await e.data.arrayBuffer();
        const b64 = arrayBufferToBase64(ab);
        await window.e2e.asrChunk(b64);
      };
      mr.start(250);
      setRecording(true);
    } catch (e) {
      console.error(e);
      setMessages((m)=>[...m,{id:uid(),role:"system",content:"🎤 Mic permission or device error"}]);
    }
  };

  const stopRecording = async () => {
    if (!recording) return;
    try {
      mediaRef.current?.stop();
      await window.e2e.asrEnd();
    } catch {}
    setRecording(false);
  };

  const canSend = input.trim().length > 0 && !streaming;
  const lastAssistant = useMemo(() => [...messages].reverse().find((m) => m.role === "assistant"), [messages]);

  return (
    <div className="h-screen w-full bg-zinc-950 text-zinc-100 flex flex-col">
      <div className="flex items-center justify-between px-4 py-3 border-b border-zinc-800">
        <div className="flex items-center gap-3">
          <div className="h-8 w-8 rounded-xl bg-indigo-600 grid place-items-center shadow">
            <Sparkles className="h-4 w-4" />
          </div>
          <div>
            <div className="text-sm uppercase tracking-wider text-zinc-400">Workspace</div>
            <div className="font-semibold">E2E Agent</div>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <Button title="Analyze code" onClick={onAnalyzeCode}>
            <Code className="h-4 w-4" /> Analyze code
          </Button>
          <Button title="Settings" className="hidden sm:flex">
            <Settings2 className="h-4 w-4" /> Settings
          </Button>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-3">
        {messages.length === 0 && (
          <div className="h-full grid place-items-center text-center text-zinc-400">
            <div>
              <div className="text-2xl font-semibold mb-2">Welcome 👋</div>
              <div>Type, analyze code, or press F9 to talk.</div>
            </div>
          </div>
        )}

        {messages.map((m) => (
          <motion.div key={m.id} initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }}>
            <div className={`max-w-3xl mx-auto flex items-start gap-3 ${m.role === "user" ? "flex-row-reverse" : ""}`}>
              <div className={`rounded-xl p-2 ${m.role === "user" ? "bg-indigo-600" : m.role === "assistant" ? "bg-zinc-800" : "bg-amber-900/40"}`}>
                {m.role === "user" ? <User className="h-4 w-4" /> : m.role === "assistant" ? <Bot className="h-4 w-4" /> : <Sparkles className="h-4 w-4" />}
              </div>
              <div className={`rounded-2xl px-4 py-3 border border-zinc-800/60 bg-zinc-900/50 w-full ${m.role === "user" ? "text-right" : "text-left"}`}>
                <div className="whitespace-pre-wrap leading-relaxed">{m.content}</div>
                {m.role === "assistant" && (
                  <div className="mt-2 flex gap-2 text-xs text-zinc-400">
                    <MsgCopy text={m.content} didCopy={(ok) => setCopyId(ok ? m.id : null)} copied={copyId === m.id} />
                    <button className="inline-flex items-center gap-1 hover:text-zinc-200" onClick={() => speak(m)} disabled={ttsBusy === m.id}>
                      {ttsBusy === m.id ? <Loader2 className="h-3 w-3 animate-spin" /> : <Play className="h-3 w-3" />} Speak
                    </button>
                    {m.audioUrl && <audio controls src={m.audioUrl} className="h-8" autoPlay />}
                  </div>
                )}
              </div>
            </div>
          </motion.div>
        ))}
        <div ref={endRef} />
      </div>

      <div className="border-t border-zinc-800 p-3">
        <div className="max-w-3xl mx-auto flex items-end gap-2">
          <Button
            title="F9 to talk"
            onMouseDown={() => startRecording()}
            onMouseUp={() => stopRecording()}
            onTouchStart={() => startRecording()}
            onTouchEnd={() => stopRecording()}
            className={`${recording ? "bg-red-700 hover:bg-red-700" : ""}`}
          >
            {recording ? <MicOff className="h-4 w-4" /> : <Mic className="h-4 w-4" />}
            {recording ? "Listening…" : "Push-to-talk (F9)"}
          </Button>

          <div className="flex-1">
            <Textarea
              ref={taRef}
              placeholder={recording ? "Listening…" : "Message the bot"}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey && input.trim() && !streaming) {
                  e.preventDefault();
                  sendUserMessage(input);
                }
              }}
            />
          </div>

          <Button title="Send" onClick={() => sendUserMessage(input)} disabled={!input.trim() || streaming}>
            {streaming ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />} Send
          </Button>

          {streaming && (
            <Button title="Stop" onClick={stop} className="bg-red-800 hover:bg-red-700">
              <Square className="h-4 w-4" /> Stop
            </Button>
          )}

          <Button
            title="Retry last"
            onClick={() => lastAssistant && chatCompletion(messages.filter((m) => m.id !== lastAssistant.id))}
          >
            <RefreshCw className="h-4 w-4" /> Retry
          </Button>
        </div>
      </div>
    </div>
  );
}

function MsgCopy({
  text, copied, didCopy
}: { text: string; copied?: boolean; didCopy: (ok: boolean) => void }) {
  const [busy, setBusy] = useState(false);
  return (
    <button
      className="inline-flex items-center gap-1 hover:text-zinc-200"
      onClick={async () => {
        try {
          setBusy(true);
          await navigator.clipboard.writeText(text);
          didCopy(true);
          setTimeout(() => didCopy(false), 1500);
        } catch {
          didCopy(false);
        } finally {
          setBusy(false);
        }
      }}
    >
      {copied ? <Check className="h-3 w-3" /> : busy ? <Loader2 className="h-3 w-3 animate-spin" /> : <ClipboardCopy className="h-3 w-3" />} Copy
    </button>
  );
}

function arrayBufferToBase64(buffer: ArrayBuffer): string {
  const bytes = new Uint8Array(buffer);
  let binary = "";
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode.apply(null, Array.from(bytes.subarray(i, i + chunk)) as any);
  }
  return btoa(binary);
}
