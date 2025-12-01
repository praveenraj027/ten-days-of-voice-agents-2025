"use client";

import { useSession } from "./session-provider";
import { WelcomeView } from "./welcome-view";
import  SessionView from "./session-view";

export function ViewController() {
  const { startSession, isSessionActive, setPlayerName, appConfig, playerName } = useSession();

  return !isSessionActive ? (
    <WelcomeView
      startSession={() => startSession()}
      setPlayerName={setPlayerName}
      isConnecting={isSessionActive}
    />
  ) : (
    <SessionView playerName={playerName} appConfig={appConfig} />
  );
}
