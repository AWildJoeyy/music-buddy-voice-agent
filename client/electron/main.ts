import { app, BrowserWindow, dialog, ipcMain } from "electron";
import path from "path";

let win: BrowserWindow | null = null;

function createWindow() {
  win = new BrowserWindow({
    width: 1100,
    height: 800,
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });
  if (process.env.VITE_DEV_SERVER_URL) win.loadURL(process.env.VITE_DEV_SERVER_URL);
  else win.loadFile(path.join(__dirname, "../dist/index.html"));
  win.on("closed", () => (win = null));
}

app.whenReady().then(createWindow);
app.on("activate", () => { if (BrowserWindow.getAllWindows().length === 0) createWindow(); });
app.on("window-all-closed", () => { if (process.platform !== "darwin") app.quit(); });

ipcMain.handle("pickFiles", async () => {
  const res = await dialog.showOpenDialog(win!, { properties: ["openFile", "multiSelections"] });
  return res.canceled ? [] : res.filePaths;
});
ipcMain.handle("ipc:pickAudio", async () => {
  if (!win) return [];
  const res = await dialog.showOpenDialog(win, {
    title: "Select audio files",
    properties: ["openFile", "multiSelections"],
    filters: [{ name: "Audio", extensions: ["mp3", "wav", "flac", "m4a"] }],
  });
  return res.canceled ? [] : res.filePaths;
});


// Chat/ASR/TTS handlers -> forward to your existing backend as before
ipcMain.handle("chat", (_e, payload) => win!.webContents.send("python", payload)); // adapt if you proxy differently
ipcMain.handle("stop", (_e, payload) => win!.webContents.send("python", payload));
ipcMain.handle("tts", (_e, payload) => win!.webContents.send("python", payload));
ipcMain.handle("asrStart", (_e, payload) => win!.webContents.send("python", { type: "asr_start", ...payload }));
ipcMain.handle("asrChunk", (_e, payload) => win!.webContents.send("python", { type: "asr_chunk", ...payload }));
ipcMain.handle("asrEnd", () => win!.webContents.send("python", { type: "asr_end" }));

// Generic pass-through to ipc_worker
ipcMain.handle("pythonInvoke", (_e, payload) => { win!.webContents.send("python", payload); });
