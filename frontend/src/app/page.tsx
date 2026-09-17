'use client';

import dynamic from "next/dynamic";

// Dynamically import AppShell to avoid SSR issues with three.js/canvas/zustand
const AppShell = dynamic(() => import("@/components/shell/AppShell"), {
  ssr: false,
  loading: () => (
    <div
      style={{
        height: "100vh",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        background: "#0d0d0f",
        color: "#3d8ef7",
        fontFamily: "monospace",
        fontSize: "12px",
        letterSpacing: "0.1em",
      }}
    >
      REALITY ENGINE — INITIALIZING WORKSTATION...
    </div>
  ),
});

export default function Page() {
  return <AppShell />;
}
