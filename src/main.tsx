import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import OverlayWindow from "./OverlayWindow";
import { applyTheme, getInitialTheme } from "./types";
import "./styles.css";

// Apply persisted Light/Dark theme immediately before first paint (zero flash)
applyTheme(getInitialTheme());

const isOverlay = window.location.hash === "#overlay";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>{isOverlay ? <OverlayWindow /> : <App />}</React.StrictMode>
);
