// app-config.ts
export type AppConfig = {
  sandboxId?: string;
  agentName?: string;
  isPreConnectBufferEnabled?: boolean;
  supportsChatInput?: boolean;
  supportsVideoInput?: boolean;
};

// Defaults used across the app. Server and RSC can import this safely.
export const APP_CONFIG_DEFAULTS: AppConfig = {
  sandboxId: undefined,
  agentName: "improv-battle-agent",
  isPreConnectBufferEnabled: true,
  supportsChatInput: true,
  supportsVideoInput: false,
};
