'use client';
import React from 'react';
import { Terminal } from 'lucide-react';

interface TerminalLogsProps {
  logs: string[];
}

export default function TerminalLogs({ logs }: TerminalLogsProps) {
  return (
    <div className="bg-zinc-900/90 border border-zinc-800 rounded-xl p-4 font-mono text-xs shadow-2xl">
      <div className="flex items-center justify-between pb-3 border-b border-zinc-800/80 mb-3">
        <div className="flex items-center gap-2">
          <Terminal className="w-4 h-4 text-emerald-400" />
          <span className="text-zinc-400 font-medium">REAL-TIME RECOVERY LOG</span>
        </div>
        <div className="flex gap-1.5">
          <div className="w-2.5 h-2.5 rounded-full bg-red-500/80" />
          <div className="w-2.5 h-2.5 rounded-full bg-amber-500/80" />
          <div className="w-2.5 h-2.5 rounded-full bg-emerald-500/80" />
        </div>
      </div>
      <div className="space-y-1.5 text-zinc-300 max-h-48 overflow-y-auto">
        {logs.length === 0 ? (
          <p className="text-zinc-500">apexcuda@t4:~$ solver daemon attached — awaiting disruption event.</p>
        ) : (
          logs.map((log, index) => (
            <p key={index} className="leading-relaxed">
              <span className="text-emerald-400">➜</span> {log}
            </p>
          ))
        )}
      </div>
    </div>
  );
}