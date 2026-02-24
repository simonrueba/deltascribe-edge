"use client";

import { useRef, useState, useCallback, useEffect } from "react";
import type { Finding } from "@/types/api";

const DELTA_COLORS: Record<string, string> = {
  improved: "#059669",
  stable: "#d97706",
  worsened: "#dc2626",
  uncertain: "#6b7280",
};

function BBoxOverlay({
  findings,
  highlighted,
  w,
  h,
}: {
  findings: Finding[];
  highlighted: number | null;
  w: number;
  h: number;
}) {
  const s = Math.min(w / 512, h / 512);
  return (
    <svg className="absolute inset-0 pointer-events-none" width={w} height={h}>
      {findings.map((f, i) => {
        if (!f.bounding_box) return null;
        const b = f.bounding_box;
        const color = DELTA_COLORS[f.delta] ?? "#6b7280";
        const active = highlighted === i;
        return (
          <g key={i} className="bbox-animate" style={{ animationDelay: `${i * 0.1}s` }}>
            <rect
              x={b.x * s} y={b.y * s} width={b.w * s} height={b.h * s}
              fill={active ? `${color}15` : "none"}
              stroke={color}
              strokeWidth={active ? 2.5 : 1.5}
              opacity={active ? 1 : 0.6}
            />
            {active && (
              <text
                x={b.x * s + 4} y={b.y * s - 5}
                fill={color} fontSize={11} fontFamily="var(--font-mono)" fontWeight={500}
              >
                {f.label}
              </text>
            )}
          </g>
        );
      })}
    </svg>
  );
}

function EmptyState() {
  return (
    <div className="absolute inset-0 flex items-center justify-center">
      <svg className="absolute inset-0 w-full h-full opacity-[0.04]">
        <defs>
          <pattern id="grid" width="48" height="48" patternUnits="userSpaceOnUse">
            <path d="M 48 0 L 0 0 0 48" fill="none" stroke="#94a3b8" strokeWidth="0.5" />
          </pattern>
        </defs>
        <rect width="100%" height="100%" fill="url(#grid)" />
      </svg>
      <div className="relative w-12 h-12">
        <div className="absolute left-1/2 top-0 bottom-0 w-px bg-slate-700/30" />
        <div className="absolute top-1/2 left-0 right-0 h-px bg-slate-700/30" />
        <div className="absolute left-1/2 top-1/2 w-1.5 h-1.5 -translate-x-1/2 -translate-y-1/2 rounded-full border border-slate-600/40" />
      </div>
    </div>
  );
}

function Panel({
  src,
  label,
  sublabel,
  findingLabel,
  findings,
  highlighted,
  scanning,
}: {
  src: string | null;
  label: string;
  sublabel?: string;
  findingLabel?: string;
  findings: Finding[];
  highlighted: number | null;
  scanning: boolean;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const [dims, setDims] = useState({ w: 0, h: 0 });

  const measure = useCallback(() => {
    if (ref.current) setDims({ w: ref.current.offsetWidth, h: ref.current.offsetHeight });
  }, []);

  useEffect(() => {
    measure();
    window.addEventListener("resize", measure);
    return () => window.removeEventListener("resize", measure);
  }, [measure]);

  return (
    <div className="flex-1 flex flex-col min-w-0 min-h-0 overflow-hidden">
      <div className="px-3 py-1.5 flex items-center gap-2 shrink-0 bg-slate-900">
        <span className="text-[11px] font-medium uppercase tracking-wider text-slate-400">
          {label}
        </span>
        {sublabel && (
          <span className="text-[10px] font-mono text-slate-500">
            {sublabel}
          </span>
        )}
        {findingLabel && (
          <span className="ml-auto text-[10px] text-slate-400 truncate max-w-[200px]" title={findingLabel}>
            {findingLabel}
          </span>
        )}
      </div>
      <div
        ref={ref}
        className={`flex-1 relative flex items-center justify-center bg-slate-950 min-h-0 overflow-hidden ${scanning ? "scanning" : ""}`}
      >
        {src ? (
          <>
            <img
              src={`data:image/png;base64,${src}`}
              alt={label}
              className="max-w-full max-h-full object-contain"
              onLoad={measure}
              draggable={false}
            />
            {findings.length > 0 && dims.w > 0 && (
              <BBoxOverlay findings={findings} highlighted={highlighted} w={dims.w} h={dims.h} />
            )}
          </>
        ) : (
          <EmptyState />
        )}
      </div>
    </div>
  );
}

interface ImageViewerProps {
  priorImage: string | null;
  currentImage: string | null;
  priorDate?: string;
  currentDate?: string;
  priorFinding?: string;
  currentFinding?: string;
  findings: Finding[];
  highlighted: number | null;
  analyzing: boolean;
}

export function ImageViewer({
  priorImage,
  currentImage,
  priorDate,
  currentDate,
  priorFinding,
  currentFinding,
  findings,
  highlighted,
  analyzing,
}: ImageViewerProps) {
  return (
    <div className="flex-1 flex min-h-0 rounded-md overflow-hidden border border-slate-200 shadow-sm">
      <Panel src={priorImage} label="Prior" sublabel={priorDate} findingLabel={priorFinding} findings={[]} highlighted={null} scanning={false} />
      <div className="w-px bg-slate-800 self-stretch" />
      <Panel src={currentImage} label="Current" sublabel={currentDate} findingLabel={currentFinding} findings={findings} highlighted={highlighted} scanning={analyzing} />
    </div>
  );
}
