'use client';
import React from 'react';
import { Plane, AlertTriangle, CheckCircle2 } from 'lucide-react';

interface AviationMapProps {
  isRecovered: boolean;
}

export default function AviationMap({ isRecovered }: AviationMapProps) {
  const hubs = [
    { id: 'DEL', name: 'Delhi', x: 260, y: 120 },
    { id: 'BOM', name: 'Mumbai', x: 170, y: 270 },
    { id: 'BLR', name: 'Bengaluru', x: 230, y: 380 },
    { id: 'HYD', name: 'Hyderabad', x: 260, y: 290 },
    { id: 'CCU', name: 'Kolkata', x: 420, y: 200 },
  ];

  return (
    <div className="relative w-full h-[400px] bg-zinc-900/60 border border-zinc-800/80 rounded-xl p-4 overflow-hidden flex flex-col justify-between">
      <div className="flex items-center justify-between z-10">
        <div>
          <h3 className="text-sm font-semibold text-zinc-200">Domestic Network Topology</h3>
          <p className="text-xs text-zinc-400">500 FLIGHTS · 5 HUBS · LIVE VECTOR FEED</p>
        </div>
        <div className="flex items-center gap-3 text-xs">
          <span className="flex items-center gap-1 text-emerald-400">
            <span className="w-2 h-2 rounded-full bg-emerald-400 inline-block" /> ON-TIME
          </span>
          <span className="flex items-center gap-1 text-red-400">
            <span className="w-2 h-2 rounded-full bg-red-400 inline-block" /> DELAYED
          </span>
        </div>
      </div>

      {/* SVG Map of India Routes */}
      <svg className="absolute inset-0 w-full h-full" viewBox="0 0 500 450">
        {/* Flight Routes */}
        <line x1="260" y1="120" x2="170" y2="270" stroke={isRecovered ? '#10b981' : '#ef4444'} strokeWidth="2" strokeDasharray="4 4" className="animate-pulse" />
        <line x1="260" y1="120" x2="230" y2="380" stroke={isRecovered ? '#10b981' : '#ef4444'} strokeWidth="2" strokeDasharray="4 4" className="animate-pulse" />
        <line x1="260" y1="120" x2="260" y2="290" stroke={isRecovered ? '#10b981' : '#10b981'} strokeWidth="1.5" strokeDasharray="4 4" />
        <line x1="260" y1="120" x2="420" y2="200" stroke={isRecovered ? '#10b981' : '#ef4444'} strokeWidth="2" strokeDasharray="4 4" />
        <line x1="170" y1="270" x2="230" y2="380" stroke="#10b981" strokeWidth="1.5" strokeDasharray="4 4" />
        <line x1="230" y1="380" x2="420" y2="200" stroke="#10b981" strokeWidth="1.5" strokeDasharray="4 4" />

        {/* Airport Hub Nodes */}
        {hubs.map((hub) => (
          <g key={hub.id} transform={`translate(${hub.x}, ${hub.y})`}>
            <circle r="6" className={hub.id === 'DEL' && !isRecovered ? 'fill-red-500 animate-ping' : 'fill-cyan-400'} />
            <circle r="4" className={hub.id === 'DEL' && !isRecovered ? 'fill-red-500' : 'fill-cyan-400'} />
            <text x="10" y="4" fill="#e2e8f0" fontSize="11" fontWeight="bold" fontFamily="monospace">
              {hub.id}
            </text>
            <text x="10" y="16" fill="#94a3b8" fontSize="9">
              {hub.name}
            </text>
          </g>
        ))}
      </svg>

      <div className="z-10 bg-zinc-950/80 backdrop-blur border border-zinc-800 rounded-lg p-2.5 flex items-center justify-between text-xs">
        <div className="flex items-center gap-2">
          {isRecovered ? (
            <>
              <CheckCircle2 className="w-4 h-4 text-emerald-400" />
              <span className="text-emerald-400 font-medium">All sectors nominal. Standby aircraft deployed.</span>
            </>
          ) : (
            <>
              <AlertTriangle className="w-4 h-4 text-red-400" />
              <span className="text-red-400 font-medium">42 Departures Stalled at DEL Hub (Dense Fog).</span>
            </>
          )}
        </div>
        <span className="text-zinc-500 font-mono">0 DGCA VIOLATIONS</span>
      </div>
    </div>
  );
}