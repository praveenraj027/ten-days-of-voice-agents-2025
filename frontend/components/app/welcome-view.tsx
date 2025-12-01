// src/welcome-view.tsx
"use client";

import { useState } from "react";

type WelcomeViewProps = {
  startSession: () => void;
  setPlayerName: (name: string) => void;
  isConnecting: boolean;
};

export function WelcomeView({
  startSession,
  setPlayerName,
  isConnecting,
}: WelcomeViewProps) {
  const [name, setName] = useState("");

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim() || isConnecting) return;

    setPlayerName(name.trim());
    startSession();
  };

  return (
    <div className="flex min-h-screen flex-col items-center justify-center gap-6 px-4 text-center">
      <h1 className="text-3xl font-bold">🎭 Improv Battle</h1>
      <p className="text-sm text-muted-foreground max-w-xs">
        Enter your stage name and join the voice-first improv game show!
      </p>

      <form onSubmit={handleSubmit} className="flex flex-col gap-4 w-full max-w-sm">
        <input
          className="w-full rounded-xl border px-3 py-2 text-sm"
          placeholder="Enter your contestant name..."
          value={name}
          onChange={(e) => setName(e.target.value)}
        />

        <button
          type="submit"
          disabled={!name.trim() || isConnecting}
          className="rounded-xl border px-4 py-2 font-semibold"
        >
          {isConnecting ? "Connecting..." : "Start Improv Battle"}
        </button>
      </form>
    </div>
  );
}
