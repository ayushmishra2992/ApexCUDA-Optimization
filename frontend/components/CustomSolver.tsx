'use client';
import React, { useState } from 'react';
import { UploadCloud, CheckCircle2, Zap } from 'lucide-react';

export default function CustomSolver() {
  const [rho, setRho] = useState(1.0);
  const [maxIter, setMaxIter] = useState(200);
  const [solved, setSolved] = useState(false);

  const handleSolve = () => {
    setSolved(true);
  };

  return (
    <div className="space-y-6">
      {/* Upload Zone */}
      <div className="border-2 border-dashed border-zinc-800 hover:border-emerald-500/60 bg-zinc-900/40 rounded-xl p-8 text-center space-y-3 transition-colors cursor-pointer">
        <UploadCloud className="w-10 h-10 text-cyan-400 mx-auto" />
        <div>
          <p className="text-sm font-medium text-zinc-200">Drop your .mps or .lp model here</p>
          <p className="text-xs text-zinc-500">or click to browse — parsed locally, solved on GPU</p>
        </div>
      </div>

      {/* Quick Load Buttons */}
      <div className="flex flex-wrap gap-2 text-xs">
        <span className="text-zinc-500 py-1">Quick Load:</span>
        <button onClick={() => setSolved(false)} className="bg-zinc-800 hover:bg-zinc-700 text-zinc-300 px-3 py-1 rounded-md border border-zinc-700 font-mono">
          📄 Netlib_AFIRO.mps
        </button>
        <button onClick={() => setSolved(false)} className="bg-zinc-800 hover:bg-zinc-700 text-zinc-300 px-3 py-1 rounded-md border border-zinc-700 font-mono">
          📄 IOCL_Refinery.mps
        </button>
        <button onClick={() => setSolved(false)} className="bg-zinc-800 hover:bg-zinc-700 text-zinc-300 px-3 py-1 rounded-md border border-zinc-700 font-mono">
          📄 ISRO_Trajectory.mps
        </button>
      </div>

      {/* PDHG Parameters */}
      <div className="bg-zinc-900/60 border border-zinc-800 rounded-xl p-5 space-y-4">
        <h4 className="text-sm font-semibold text-zinc-200">PDHG Hyperparameters</h4>
        
        <div className="space-y-2">
          <div className="flex justify-between text-xs font-mono">
            <span className="text-zinc-400">PENALTY RHO (ρ):</span>
            <span className="text-emerald-400 font-bold">{rho}</span>
          </div>
          <input
            type="range"
            min="0.1"
            max="5.0"
            step="0.1"
            value={rho}
            onChange={(e) => setRho(parseFloat(e.target.value))}
            className="w-full accent-emerald-500"
          />
        </div>

        <div className="space-y-2">
          <div className="flex justify-between text-xs font-mono">
            <span className="text-zinc-400">MAX ITERATIONS:</span>
            <span className="text-emerald-400 font-bold">{maxIter}</span>
          </div>
          <input
            type="range"
            min="50"
            max="500"
            step="50"
            value={maxIter}
            onChange={(e) => setMaxIter(parseInt(e.target.value))}
            className="w-full accent-emerald-500"
          />
        </div>

        <button
          onClick={handleSolve}
          className="w-full bg-gradient-to-r from-cyan-500 to-emerald-500 hover:from-cyan-400 hover:to-emerald-400 text-zinc-950 font-bold py-3 rounded-lg flex items-center justify-center gap-2 text-sm shadow-lg shadow-emerald-950/40"
        >
          <Zap className="w-4 h-4 fill-current" /> Solve with ApexCUDA GPU
        </button>
      </div>

      {/* JSON Response Terminal */}
      {solved && (
        <div className="bg-zinc-900 border border-emerald-500/40 rounded-xl p-4 font-mono text-xs space-y-2">
          <div className="flex items-center justify-between text-emerald-400">
            <span className="flex items-center gap-1.5"><CheckCircle2 className="w-4 h-4" /> SOLVER RESPONSE · JSON</span>
            <span className="text-zinc-500">LATENCY: 0.034s</span>
          </div>
          <pre className="text-zinc-300 overflow-x-auto p-2 bg-zinc-950 rounded">
{JSON.stringify({
  status: "OPTIMAL_CONVERGENCE",
  gpu_time_ms: 34.2,
  pdhg_iterations: 62,
  primal_residual: "8.42e-5",
  dual_residual: "6.10e-5",
  objective_value: 12860000.0,
  dgca_violations: 0
}, null, 2)}
          </pre>
        </div>
      )}
    </div>
  );
}