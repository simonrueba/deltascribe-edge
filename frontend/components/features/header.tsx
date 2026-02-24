"use client";

import { useEffect, useState } from "react";
import { healthCheck } from "@/lib/api";
import type { HealthResponse } from "@/types/api";

export function Header() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [offline, setOffline] = useState(false);

  useEffect(() => {
    let mounted = true;
    const check = async () => {
      try {
        const h = await healthCheck();
        if (mounted) { setHealth(h); setOffline(false); }
      } catch {
        if (mounted) setOffline(true);
      }
    };
    check();
    const id = setInterval(check, 20000);
    return () => { mounted = false; clearInterval(id); };
  }, []);

  return (
    <header className="h-14 flex items-center justify-between px-5 border-b border-slate-200 bg-white select-none shrink-0">
      <div className="flex items-center gap-3">
        <div className="w-8 h-8 rounded-md bg-primary flex items-center justify-center">
          <span className="text-white text-sm font-bold">DS</span>
        </div>
        <div>
          <span className="text-[15px] font-semibold tracking-tight text-slate-900">
            DeltaScribe
          </span>
          <span className="text-[11px] text-slate-400 ml-1.5">
            Edge
          </span>
        </div>
      </div>

      <div className="flex items-center gap-5 text-[13px] text-slate-500">
        {health && !health.model_loaded && (
          <span className="text-amber-600 text-[12px]">
            Model not loaded
          </span>
        )}
        <div className="flex items-center gap-2">
          <div
            className={`w-2 h-2 rounded-full ${
              offline ? "bg-red-500" : "bg-emerald-500"
            }`}
          />
          <span className={offline ? "text-red-600" : "text-slate-500"}>
            {offline ? "Offline" : "Connected"}
          </span>
        </div>
      </div>
    </header>
  );
}
