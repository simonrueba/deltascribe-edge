"use client";

import { useCallback, useState, useRef } from "react";

interface UploadDialogProps {
  open: boolean;
  onClose: () => void;
  onUpload: (prior: string, current: string) => void;
}

function fileToBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve((reader.result as string).split(",")[1]);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

export function UploadDialog({ open, onClose, onUpload }: UploadDialogProps) {
  const [prior, setPrior] = useState<string | null>(null);
  const [current, setCurrent] = useState<string | null>(null);
  const [priorName, setPriorName] = useState("");
  const [currentName, setCurrentName] = useState("");
  const priorRef = useRef<HTMLInputElement>(null);
  const currentRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback(
    async (file: File, target: "prior" | "current") => {
      if (!file.type.startsWith("image/")) return;
      const b64 = await fileToBase64(file);
      if (target === "prior") { setPrior(b64); setPriorName(file.name); }
      else { setCurrent(b64); setCurrentName(file.name); }
    },
    [],
  );

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40" onClick={onClose}>
      <div
        className="bg-white border border-slate-200 shadow-lg p-5 w-[380px] rounded-md"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="text-[10px] font-mono uppercase tracking-[0.15em] text-slate-400 mb-4">
          Upload CXR Pair
        </div>

        <div className="space-y-3">
          {(["prior", "current"] as const).map((target) => {
            const name = target === "prior" ? priorName : currentName;
            const ref = target === "prior" ? priorRef : currentRef;
            return (
              <div key={target}>
                <button
                  onClick={() => ref.current?.click()}
                  className="w-full text-left px-3 py-2.5 border border-slate-200 rounded-sm hover:border-slate-300 hover:bg-slate-50 transition-colors"
                >
                  <div className="text-[10px] font-mono uppercase tracking-wider text-slate-400">
                    {target} CXR
                  </div>
                  <div className="text-[11px] text-slate-600 mt-0.5 truncate">
                    {name || "Click to select..."}
                  </div>
                </button>
                <input
                  ref={ref}
                  type="file"
                  accept="image/png,image/jpeg"
                  className="hidden"
                  onChange={(e) => {
                    const f = e.target.files?.[0];
                    if (f) handleFile(f, target);
                  }}
                />
              </div>
            );
          })}
        </div>

        <div className="flex justify-end gap-2 mt-4">
          <button
            onClick={onClose}
            className="text-[10px] font-mono text-slate-500 hover:text-slate-700 px-3 py-1.5 transition-colors"
          >
            CANCEL
          </button>
          <button
            disabled={!prior || !current}
            onClick={() => {
              if (prior && current) {
                onUpload(prior, current);
                onClose();
              }
            }}
            className="text-[10px] font-mono bg-primary text-white px-3 py-1.5 rounded-sm disabled:opacity-30 hover:bg-blue-900 transition-colors"
          >
            LOAD
          </button>
        </div>
      </div>
    </div>
  );
}
