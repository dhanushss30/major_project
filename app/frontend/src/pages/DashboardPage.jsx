import { useEffect, useState } from 'react'
import {
  ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid,
  LineChart, Line, ReferenceLine, ScatterChart, Scatter, Cell, AreaChart, Area
} from 'recharts'
import {
  Activity, TrendingUp, Layers, Cpu, BarChart3 as B3, Sparkles
} from 'lucide-react'
import {
  getMetricsSummary, getPerClassAUC, getTrainingTrajectory
} from '../api/client'

export default function DashboardPage() {
  const [summary,    setSummary]    = useState(null)
  const [perClass,   setPerClass]   = useState(null)
  const [trajectory, setTrajectory] = useState(null)
  const [loading,    setLoading]    = useState(true)

  useEffect(() => {
    Promise.all([getMetricsSummary(), getPerClassAUC(), getTrainingTrajectory()])
      .then(([s, p, t]) => {
        setSummary(s); setPerClass(p); setTrajectory(t)
      })
      .catch(console.error)
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <Loading />

  return (
    <div className="space-y-8">
      <div>
        <div className="heading-mono mb-2">// Dashboard</div>
        <h1 className="heading-display text-4xl">Model performance metrics</h1>
        <p className="text-muted-light mt-2">
          Live readout of the 4-checkpoint ensemble on the fold-0 validation set.
        </p>
      </div>

      {/* ─── Headline stats ──────────────────────────────────────────── */}
      {summary && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <Stat label="Macro ROC AUC" value={summary.ensemble_val_auc.toFixed(4)} accent="cyan" big />
          <Stat label="Checkpoints"   value={summary.n_ckpts} accent="lime" />
          <Stat label="Classes"        value={summary.n_classes} accent="amber" />
          <Stat label="Well-trained"  value={summary.n_well_trained} accent="violet" />
        </div>
      )}

      {/* ─── Ensemble vs single ───────────────────────────────────────── */}
      {summary && (
        <div className="card p-6">
          <div className="flex items-center gap-2 mb-4">
            <Layers className="w-5 h-5 text-accent-cyan" />
            <div className="heading-mono">// Ensemble vs single-checkpoint AUC</div>
          </div>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart
              data={[
                { name: 'v3 clean',  auc: summary.single_aucs.v3_fold0_clean },
                { name: 'v3 FiLM',   auc: summary.single_aucs.v3_fold0_film  },
                { name: 'ESC-50 BG', auc: summary.single_aucs.esc50_bg_peak  },
                { name: 'fold 1',    auc: summary.single_aucs.new_fold1     },
                { name: 'Ensemble + TTA', auc: summary.ensemble_val_auc },
              ]}
              margin={{ top: 10, right: 10, left: 0, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#2a3340" />
              <XAxis dataKey="name" stroke="#7a8290" fontSize={12} />
              <YAxis stroke="#7a8290" domain={[0.6, 0.85]} fontSize={12} />
              <Tooltip
                contentStyle={{ background: '#161b22', border: '1px solid #2a3340', borderRadius: 6 }}
                cursor={{ fill: 'rgba(0,240,255,0.04)' }}
              />
              <ReferenceLine y={0.82} stroke="#a3ff00" strokeDasharray="3 3" label={{ value: 'Target 0.82', fill: '#a3ff00', fontSize: 11 }} />
              <Bar dataKey="auc" radius={[6, 6, 0, 0]}>
                {[
                  { name: 'v3 clean',  fill: '#3b4452' },
                  { name: 'v3 FiLM',   fill: '#3b4452' },
                  { name: 'ESC-50 BG', fill: '#3b4452' },
                  { name: 'fold 1',    fill: '#3b4452' },
                  { name: 'Ensemble',  fill: '#00f0ff' },
                ].map((d, i) => <Cell key={i} fill={d.fill} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* ─── Training trajectory ─────────────────────────────────────── */}
      {trajectory?.available && (
        <div className="card p-6">
          <div className="flex items-center gap-2 mb-4">
            <TrendingUp className="w-5 h-5 text-accent-lime" />
            <div className="heading-mono">// Training trajectory · fold 1</div>
          </div>
          <ResponsiveContainer width="100%" height={280}>
            <AreaChart data={trajectory.rows} margin={{ top: 10, right: 10, left: 0, bottom: 5 }}>
              <defs>
                <linearGradient id="gAuc" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%"   stopColor="#a3ff00" stopOpacity={0.4} />
                  <stop offset="100%" stopColor="#a3ff00" stopOpacity={0}   />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#2a3340" />
              <XAxis dataKey="epoch" stroke="#7a8290" fontSize={12} />
              <YAxis domain={[0.45, 0.75]} stroke="#7a8290" fontSize={12} />
              <Tooltip
                contentStyle={{ background: '#161b22', border: '1px solid #2a3340', borderRadius: 6 }}
              />
              <Area
                type="monotone" dataKey="val_auc"
                stroke="#a3ff00" strokeWidth={2}
                fill="url(#gAuc)"
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* ─── Per-class AUC distribution ──────────────────────────────── */}
      {perClass?.available && (
        <>
          <div className="card p-6">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2">
                <B3 className="w-5 h-5 text-accent-cyan" />
                <div className="heading-mono">// Per-class AUC distribution</div>
              </div>
              <div className="font-mono text-xs text-muted">
                mean {perClass.mean_auc.toFixed(3)} · median {perClass.median_auc.toFixed(3)} · {perClass.n_valid}/{perClass.n_total} valid
              </div>
            </div>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={perClass.rows.map((r, i) => ({ idx: i, ...r }))}>
                <CartesianGrid strokeDasharray="3 3" stroke="#2a3340" />
                <XAxis dataKey="idx" stroke="#7a8290" fontSize={10}
                       label={{ value: 'class rank', fill: '#7a8290', fontSize: 10, offset: -5, position: 'insideBottom' }} />
                <YAxis stroke="#7a8290" fontSize={10} domain={[0, 1]} />
                <Tooltip
                  content={({ payload }) => {
                    if (!payload || !payload[0]) return null
                    const d = payload[0].payload
                    return (
                      <div className="card px-3 py-2 text-xs">
                        <div className="font-mono text-accent-cyan">{d.code}</div>
                        {d.common_name && <div className="text-muted">{d.common_name}</div>}
                        <div className="mt-1">AUC: <span className="font-mono text-white">{d.auc.toFixed(3)}</span></div>
                        <div>train samples: <span className="font-mono text-white">{d.n_train}</span></div>
                      </div>
                    )
                  }}
                />
                <Bar dataKey="auc" radius={[3, 3, 0, 0]}>
                  {perClass.rows.map((r, i) => (
                    <Cell key={i} fill={
                      r.auc > 0.9 ? '#a3ff00'
                      : r.auc > 0.8 ? '#00f0ff'
                      : r.auc > 0.6 ? '#ffb800'
                      : '#ff6b6b'
                    } />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            <div className="flex gap-3 mt-3 text-xs">
              <Legend color="#a3ff00" label="AUC > 0.90" />
              <Legend color="#00f0ff" label="0.80 – 0.90" />
              <Legend color="#ffb800" label="0.60 – 0.80" />
              <Legend color="#ff6b6b" label="< 0.60" />
            </div>
          </div>

          <div className="card p-6">
            <div className="flex items-center gap-2 mb-4">
              <Activity className="w-5 h-5 text-accent-violet" />
              <div className="heading-mono">// AUC vs training-sample count</div>
            </div>
            <ResponsiveContainer width="100%" height={320}>
              <ScatterChart margin={{ top: 10, right: 10, left: 0, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#2a3340" />
                <XAxis
                  type="number" dataKey="n_train"
                  stroke="#7a8290" fontSize={10}
                  scale="log" domain={[1, 'auto']}
                  label={{ value: 'training samples (log)', fill: '#7a8290', fontSize: 10, position: 'insideBottom', offset: -2 }}
                />
                <YAxis
                  type="number" dataKey="auc"
                  stroke="#7a8290" fontSize={10} domain={[0, 1]}
                  label={{ value: 'AUC', fill: '#7a8290', fontSize: 10, angle: -90, position: 'insideLeft' }}
                />
                <ReferenceLine x={50} stroke="#7a8290" strokeDasharray="3 3"
                  label={{ value: 'rare-class cutoff', fill: '#7a8290', fontSize: 10 }} />
                <Tooltip
                  content={({ payload }) => {
                    if (!payload || !payload[0]) return null
                    const d = payload[0].payload
                    return (
                      <div className="card px-3 py-2 text-xs">
                        <div className="font-mono text-accent-cyan">{d.code}</div>
                        <div>AUC: {d.auc.toFixed(3)} · train: {d.n_train}</div>
                      </div>
                    )
                  }}
                />
                <Scatter data={perClass.rows} fill="#a78bfa" />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </>
      )}

      {/* ─── Ckpts table ─────────────────────────────────────────────── */}
      {summary && (
        <div className="card p-6">
          <div className="flex items-center gap-2 mb-4">
            <Cpu className="w-5 h-5 text-accent-cyan" />
            <div className="heading-mono">// Loaded checkpoints</div>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-muted-light font-mono text-xs border-b border-ink-500/40">
                  <th className="py-2">Name</th>
                  <th>Arch</th>
                  <th>val_auc</th>
                  <th>Epoch</th>
                </tr>
              </thead>
              <tbody>
                {summary.ckpts.map((c) => (
                  <tr key={c.name} className="border-b border-ink-500/20 last:border-0">
                    <td className="py-2 font-mono text-xs">{c.name}</td>
                    <td><span className={`label-chip ${c.has_film ? 'text-accent-violet border-accent-violet/40' : ''}`}>
                      {c.has_film ? 'FiLM' : 'clean'}
                    </span></td>
                    <td className="font-mono">{c.val_auc?.toFixed(4) ?? '—'}</td>
                    <td className="font-mono text-muted">{c.epoch ?? '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}

function Stat({ label, value, accent, big = false }) {
  const colors = {
    cyan:   'text-accent-cyan',
    lime:   'text-accent-lime',
    amber:  'text-accent-amber',
    violet: 'text-accent-violet',
  }
  return (
    <div className={`card-glow p-5 ${big ? 'border-accent-cyan/40 bg-accent-cyan/5' : ''}`}>
      <div className="data-stat">{label}</div>
      <div className={`data-value mt-1 ${colors[accent]} ${big ? 'glow-text-cyan' : ''}`}>
        {value}
      </div>
    </div>
  )
}

function Legend({ color, label }) {
  return (
    <div className="flex items-center gap-1.5 text-muted">
      <span className="w-3 h-3 rounded-sm" style={{ background: color }} />
      {label}
    </div>
  )
}

function Loading() {
  return (
    <div className="space-y-4">
      <div className="card-glow h-12 shimmer" />
      <div className="grid grid-cols-4 gap-4">
        {[1,2,3,4].map(i => <div key={i} className="card h-20 shimmer" />)}
      </div>
      <div className="card h-80 shimmer" />
    </div>
  )
}
