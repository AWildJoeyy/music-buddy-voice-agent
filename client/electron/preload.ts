import { contextBridge, ipcRenderer } from "electron";

const send = (payload: any) => ipcRenderer.invoke("ipc:send", payload);

const events = {
  subscribe(handler: (msg: any) => void) {
    const cb = (_: any, data: any) => handler(data);
    ipcRenderer.on("ipc:python", cb);
    return () => ipcRenderer.removeListener("ipc:python", cb);
  },
  hotkeys(handler: (ev: any) => void) {
    const cb = (_: any, data: any) => handler(data);
    ipcRenderer.on("ipc:hotkey", cb);
    return () => ipcRenderer.removeListener("ipc:hotkey", cb);
  },
};

contextBridge.exposeInMainWorld("e2e", {
  // existing
  chat: (id: string, messages: {role:"user"|"assistant"|"system"; content:string}[], stream = true) =>
    send({ type: "chat", id, messages, stream }),
  stop: (id: string) => send({ type: "stop", id }),
  tts: (id: string, text: string) => send({ type: "tts", id, text }),
  asrStart: (fmt: "webm" | "wav" = "webm") => send({ type: "asr_start", fmt }),
  asrChunk: (data_b64: string) => send({ type: "asr_chunk", data_b64 }),
  asrEnd: () => send({ type: "asr_end" }),
  pickFiles: () => ipcRenderer.invoke("ipc:pick"),
  onPython: events.subscribe,
  onHotkey: events.hotkeys,

  // new music helpers (minimal)
  pickAudio: () => ipcRenderer.invoke("ipc:pickAudio"),
  musicAutoBucket: (paths: string[]) => send({ type: "music_auto_bucket", paths }),
  playPlaylist: (playlist: string, shuffle = false) => send({ type: "music_play_playlist", playlist, shuffle }),
});
