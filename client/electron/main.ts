import { app, BrowserWindow, ipcMain, dialog, Menu, globalShortcut } from "electron";
import path from "node:path";
import { spawn, ChildProcessWithoutNullStreams } from "node:child_process";
import fs from "node:fs";

let win: BrowserWindow | null = null;
let py: ChildProcessWithoutNullStreams | null = null;

function attachPythonHandlers() {
  if (!py || !win) return;
  const sendToRenderer = (data: any) => win && win.webContents.send("ipc:python", data);

  py.stdout.setEncoding("utf8");
  py.stdout.on("data", (chunk) => {
    for (const line of String(chunk).split("\n")) {
      const t = line.trim();
      if (!t) continue;
      try { sendToRenderer(JSON.parse(t)); }
      catch { /* ignore bad lines */ }
    }
  });

  py.stderr.setEncoding("utf8");
  py.stderr.on("data", (chunk) => {
    sendToRenderer({ type: "stderr", data: String(chunk) });
  });

  py.on("exit", (code) => {
    sendToRenderer({ type: "stderr", data: `python exited (${code})` });
  });
}

function startPython() {
  const appSrc = path.join(process.cwd(), "..", "app", "src");
  const env = {
    ...process.env,
    PYTHONPATH: [appSrc, process.env.PYTHONPATH || ""].filter(Boolean).join(path.delimiter),
  };
  py = spawn("python", ["-m", "voice_agent.ipc_worker"], {
    cwd: appSrc,
    env,
    stdio: ["pipe", "pipe", "pipe"],
  });
  attachPythonHandlers();
}

async function createWindow() {
  win = new BrowserWindow({
    width: 1120,
    height: 760,
    backgroundColor: "#0a0a0a",
    title: "E2E Agent (IPC)",
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
    },
  });

  if (process.env.VITE_DEV) {
    await win.loadURL("http://localhost:5173");
    win.webContents.openDevTools({ mode: "detach" });
  } else {
    const distIndex = path.join(__dirname, "../dist/index.html");
    if (fs.existsSync(distIndex)) {
      await win.loadURL(new URL(`file://${distIndex}`).toString());
    } else {
      await win.loadURL("http://localhost:5173");
      win.webContents.openDevTools({ mode: "detach" });
    }
  }

  // Global shortcut: F9 toggles push-to-talk
  app.whenReady().then(() => {
    globalShortcut.register("F9", () => {
      if (win && !win.isDestroyed()) {
        win.webContents.send("ipc:hotkey", { type: "ptt-toggle" });
      }
    });
  });

  win.on("closed", () => {
    globalShortcut.unregisterAll();
    win = null;
    if (py) { py.kill(); py = null; }
  });
}

ipcMain.handle("ipc:send", async (_event, payload) => {
  if (!py) return;
  py.stdin.write(JSON.stringify(payload) + "\n");
});

ipcMain.handle("ipc:pick", async () => {
  if (!win) return [];
  const res = await dialog.showOpenDialog(win, {
    properties: ["openFile", "openDirectory", "multiSelections"],
  });
  return res.canceled ? [] : res.filePaths;
});

app.whenReady().then(() => {
  createWindow();
  startPython();
  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on("will-quit", () => {
  globalShortcut.unregisterAll();
});
