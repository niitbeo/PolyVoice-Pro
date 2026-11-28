import React, { useEffect, useState } from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import OverlayWindow from "./OverlayWindow";
import { applyTheme, getInitialTheme } from "./types";
import "./styles.css";

function isOverlayLocation(): boolean {
  const h = window.location.hash || "";
  if (h === "#overlay" || h.startsWith("#overlay?")) return true;
  const params = new URLSearchParams(window.location.search || "");
  return params.get("window") === "overlay";
}

// Apply persisted Light/Dark theme immediately before first paint (zero flash)
applyTheme(getInitialTheme());
if (isOverlayLocation()) {
  document.documentElement.classList.add("overlay-body");
  document.body.classList.add("overlay-body");
}

function RootView() {
  const [isOverlay, setIsOverlay] = useState(() => isOverlayLocation());

  useEffect(() => {
    const onHashChange = () => {
      const next = isOverlayLocation();
      setIsOverlay(next);
      document.documentElement.classList.toggle("overlay-body", next);
      document.body.classList.toggle("overlay-body", next);
    };
    window.addEventListener("hashchange", onHashChange);
    return () => window.removeEventListener("hashchange", onHashChange);
  }, []);

  return isOverlay ? <OverlayWindow /> : <App />;
}

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <RootView />
  </React.StrictMode>
);
