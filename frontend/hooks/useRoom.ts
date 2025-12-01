// src/hooks/useRoom.ts
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Room, RoomEvent, TokenSource } from 'livekit-client';
import { AppConfig } from '@/app-config';
import { toastAlert } from '@/components/livekit/alert-toast';

export function useRoom(appConfig: AppConfig, playerName?: string) {
  const aborted = useRef(false);
  const room = useMemo(() => new Room(), []);
  const [isSessionActive, setIsSessionActive] = useState(false);

  useEffect(() => {
    function onDisconnected() {
      setIsSessionActive(false);
    }

    function onMediaDevicesError(error: Error) {
      toastAlert({
        title: 'Encountered an error with your media devices',
        description: `${error.name}: ${error.message}`,
      });
    }

    room.on(RoomEvent.Disconnected, onDisconnected);
    room.on(RoomEvent.MediaDevicesError, onMediaDevicesError);

    return () => {
      room.off(RoomEvent.Disconnected, onDisconnected);
      room.off(RoomEvent.MediaDevicesError, onMediaDevicesError);
    };
  }, [room]);

  useEffect(() => {
    return () => {
      aborted.current = true;
      try {
        room.disconnect();
      } catch {
        /* ignore */
      }
    };
  }, [room]);

  // DEV: expose room to window for debugging (remove after debugging)
  useEffect(() => {
    if (room && typeof window !== 'undefined') {
      try {
        (window as any).room = room;
        // Make this obvious in console while debugging
        // eslint-disable-next-line no-console
        console.log('DEBUG: LiveKit room attached to window.room');
      } catch {
        /* ignore */
      }
    }
    return () => {
      try {
        if ((window as any).room === room) {
          delete (window as any).room;
        }
      } catch {}
    };
  }, [room]);

  // Token source
  const tokenSource = useMemo(
    () =>
      TokenSource.custom(async () => {
        const url = new URL(
          process.env.NEXT_PUBLIC_CONN_DETAILS_ENDPOINT ?? '/api/connection-details',
          typeof window !== 'undefined' ? window.location.origin : 'http://localhost'
        );

        try {
          const res = await fetch(url.toString(), {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
              'X-Sandbox-Id': appConfig.sandboxId ?? '',
            },
            body: JSON.stringify({
              room_config: appConfig.agentName
                ? {
                    agents: [{ agent_name: appConfig.agentName }],
                  }
                : undefined,
            }),
          });
          return await res.json();
        } catch (error) {
          // eslint-disable-next-line no-console
          console.error('Error fetching connection details:', error);
          throw new Error('Error fetching connection details!');
        }
      }),
    [appConfig]
  );

  const startSession = useCallback(() => {
    setIsSessionActive(true);

    if (room.state === 'disconnected') {
      const { isPreConnectBufferEnabled } = appConfig;

      Promise.all([
        room.localParticipant.setMicrophoneEnabled(true, undefined, {
          preConnectBuffer: isPreConnectBufferEnabled,
        }),

        tokenSource
          .fetch({ agentName: appConfig.agentName })
          .then(async (connectionDetails) => {
            // connect first (some livekit-client builds disallow metadata arg in connect)
            await room.connect(connectionDetails.serverUrl, connectionDetails.participantToken);

            // then set metadata — fail-soft
            try {
              await room.localParticipant.setMetadata(JSON.stringify({ playerName: playerName ?? 'Player' }));
            } catch (err) {
              // ignore metadata set errors
            }
          }),
      ]).catch((error) => {
        if (aborted.current) return;

        toastAlert({
          title: 'There was an error connecting to the agent',
          description: `${(error as Error).name}: ${(error as Error).message}`,
        });
      });
    }
  }, [room, appConfig, tokenSource, playerName]);

  const endSession = useCallback(() => {
    setIsSessionActive(false);
    try {
      room.disconnect();
    } catch {}
  }, [room]);

  return { room, isSessionActive, startSession, endSession };
}
