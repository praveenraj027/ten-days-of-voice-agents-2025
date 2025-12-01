"use client";

import React from "react";
import { useChatMessages } from "@/hooks/useChatMessages";
import { AgentControlBar } from "@/components/livekit/agent-control-bar/agent-control-bar";


export default function SessionViewFixed({ playerName, appConfig }: { playerName: string; appConfig?: Record<string, any> }) {
  const messages = useChatMessages();

  return (
    <div className="flex h-full w-full justify-center items-center">
      
      <div className="w-[50%] h-full p-6 text-white bg-black flex flex-col rounded-xl ">
        {/* HEADER */}
        <div className="text-2xl font-bold mb-4">
          🎭 Improv Battle — Hello {playerName}!
        </div>

        {/* DEBUG MESSAGE COUNT */}
        <div className="text-sm text-gray-300 mb-2">
          Messages received: {messages.length}
        </div>

        {/* MESSAGE BOX */}
        <div className="flex-1 bg-zinc-900 rounded-xl p-4 overflow-y-auto border border-zinc-700">
          {messages.length === 0 ? (
            <div className="text-gray-400 text-center mt-10">
              👀 Waiting for agent messages…
              <br />
              <span className="text-xs opacity-60">Say something like “Hello!”</span>
            </div>
          ) : (
            messages.map((m, i) => (
              <div key={i} className="mb-3">
                <div className="text-xs text-gray-500 mb-1">
                  {(m as any).role === "agent" ? "🤖 Agent" : "🧑 You"}
                </div>
                <div className="bg-zinc-800 border border-zinc-700 p-3 rounded-lg">
                  {(m as any).text}
                </div>
              </div>
            ))
          )}
        </div>

        {/* CONTROL BAR */}
        <div className="mt-4">
          <AgentControlBar
            controls={{
              leave: true,
              microphone: true,
              camera: false,
              screenShare: false,
              chat: false,
            }}
          />

        </div>
      </div>
    </div>
  );
}
