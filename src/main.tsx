import React, { useEffect, useState } from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import OverlayWindow from "./OverlayWindow";
import { applyTheme, getInitialTheme } from "./types";
import "./styles.css";

// Apply persisted Light/Dark theme immediately before first paint (zero flash)
applyTheme(getInitialTheme());

function RootView() {
  const [isOverlay, setIsOverlay] = useState(() => window.location.hash === "#overlay");

  useEffect(() => {
    const onHashChange = () => setIsOverlay(window.location.hash === "#overlay");
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
