import React from "react";
import { createRoot } from "react-dom/client";
import ChatWindow from "./ChatWindow";
import "./index.css";

createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <ChatWindow />
  </React.StrictMode>
);
