"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import { transcribeAudio } from "@/lib/api";

type DictationState = "idle" | "recording" | "transcribing";

interface DictationButtonProps {
  onTranscript: (text: string, audioBase64: string) => void;
  disabled?: boolean;
}

export function DictationButton({ onTranscript, disabled }: DictationButtonProps) {
  const [state, setState] = useState<DictationState>("idle");
  const [transcript, setTranscript] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const streamRef = useRef<MediaStream | null>(null);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      recorderRef.current?.stop();
      streamRef.current?.getTracks().forEach((t) => t.stop());
    };
  }, []);

  const startRecording = useCallback(async () => {
    setError(null);
    setTranscript(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      const recorder = new MediaRecorder(stream);
      recorderRef.current = recorder;
      chunksRef.current = [];

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      recorder.onstop = async () => {
        stream.getTracks().forEach((t) => t.stop());
        streamRef.current = null;

        const blob = new Blob(chunksRef.current, { type: "audio/webm" });
        const reader = new FileReader();
        reader.onloadend = async () => {
          const base64 = (reader.result as string).split(",")[1];
          setState("transcribing");
          try {
            const res = await transcribeAudio(base64);
            const text = res.text ?? "";
            setTranscript(text);
            onTranscript(text, base64);
          } catch (e) {
            const msg = e instanceof Error ? e.message : "Transcription failed";
            if (msg.includes("No ASR engine")) {
              setError("No speech engine available — check backend audio deps");
            } else {
              setError(msg);
            }
          } finally {
            setState("idle");
          }
        };
        reader.readAsDataURL(blob);
      };

      recorder.start();
      setState("recording");
    } catch {
      setError("Microphone access denied");
      setState("idle");
    }
  }, [onTranscript]);

  const stopRecording = useCallback(() => {
    recorderRef.current?.stop();
  }, []);

  const handleClick = () => {
    if (state === "recording") {
      stopRecording();
    } else if (state === "idle") {
      startRecording();
    }
  };

  return (
    <div className="flex items-center gap-2">
      <button
        onClick={handleClick}
        disabled={disabled || state === "transcribing"}
        title={
          state === "recording"
            ? "Stop recording"
            : state === "transcribing"
              ? "Transcribing..."
              : "Dictate notes"
        }
        className={`relative flex items-center justify-center w-9 h-9 rounded-md border transition-all ${
          state === "recording"
            ? "bg-red-50 border-red-300 text-red-600 hover:bg-red-100"
            : state === "transcribing"
              ? "bg-slate-50 border-slate-200 text-slate-400 cursor-wait"
              : "bg-white border-slate-200 text-slate-500 hover:text-slate-700 hover:bg-slate-50"
        } disabled:opacity-40 disabled:cursor-not-allowed`}
      >
        {state === "transcribing" ? (
          <span className="w-4 h-4 border-2 border-slate-300 border-t-slate-600 rounded-full animate-spin" />
        ) : (
          <svg
            width="16"
            height="16"
            viewBox="0 0 16 16"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
          >
            <path
              d="M8 1a2.5 2.5 0 0 0-2.5 2.5v4a2.5 2.5 0 0 0 5 0v-4A2.5 2.5 0 0 0 8 1Z"
              fill="currentColor"
            />
            <path
              d="M3.5 6.5a.5.5 0 0 1 1 0v1a3.5 3.5 0 1 0 7 0v-1a.5.5 0 0 1 1 0v1a4.5 4.5 0 0 1-4 4.473V13.5h2a.5.5 0 0 1 0 1h-5a.5.5 0 0 1 0-1h2v-1.527a4.5 4.5 0 0 1-4-4.473v-1Z"
              fill="currentColor"
            />
          </svg>
        )}
        {state === "recording" && (
          <span className="absolute -top-0.5 -right-0.5 w-2.5 h-2.5 bg-red-500 rounded-full animate-pulse" />
        )}
      </button>

      {state === "transcribing" && (
        <span className="text-[12px] text-slate-400">Transcribing...</span>
      )}
      {transcript && state === "idle" && (
        <span className="text-[12px] text-emerald-600">Transcribed</span>
      )}
      {error && (
        <span className="text-[12px] text-red-500">{error}</span>
      )}
    </div>
  );
}
