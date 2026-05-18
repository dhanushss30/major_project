import { Link } from 'react-router-dom'
import { Award, Info, ChevronRight } from 'lucide-react'

export default function SpeciesCard({ species, rank, variant = 'row' }) {
  const pct = species.prob * 100
  const isBird = species.is_bird !== false

  if (variant === 'row') {
    return (
      <div className="flex items-center gap-3 px-3 py-2.5 rounded-lg border border-ink-500/40 bg-ink-700/30 hover:border-accent-cyan/30 transition-all">
        {rank && (
          <div className={`w-6 h-6 rounded-full flex items-center justify-center font-mono text-xs flex-shrink-0
            ${rank === 1 ? 'bg-accent-cyan/20 text-accent-cyan border border-accent-cyan/40'
                          : 'bg-ink-600 text-muted-light'}`}>
            {rank}
          </div>
        )}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <Link to={`/species/${species.code}`}
                  className="font-medium hover:text-accent-cyan transition-colors truncate">
              {species.common_name || species.code}
            </Link>
            {!isBird && (
              <span className="label-chip text-accent-amber border-accent-amber/40">non-bird</span>
            )}
            {species.well_trained && (
              <Award className="w-3 h-3 text-accent-lime" />
            )}
          </div>
          <div className="font-mono text-[11px] text-muted">{species.code}</div>
          <div className="mt-1.5">
            <div className="conf-bar">
              <div
                className="conf-bar-fill"
                style={{ width: `${Math.min(100, pct * 3)}%` }}
              />
            </div>
          </div>
        </div>
        <div className="text-right">
          <div className="font-mono text-sm font-semibold text-accent-cyan">
            {species.prob.toFixed(3)}
          </div>
          <div className="text-[10px] text-muted">{pct.toFixed(1)}%</div>
        </div>
      </div>
    )
  }

  // overall variant — bigger card
  return (
    <Link
      to={`/species/${species.code}`}
      className="card-glow p-4 block hover:border-accent-cyan/40"
    >
      <div className="flex items-start justify-between mb-2">
        <div>
          <div className="font-display font-semibold truncate">
            {species.common_name || species.code}
          </div>
          <div className="font-mono text-xs text-muted">{species.code}</div>
        </div>
        <ChevronRight className="w-4 h-4 text-muted" />
      </div>
      <div className="conf-bar mb-1">
        <div className="conf-bar-fill" style={{ width: `${Math.min(100, pct * 3)}%` }} />
      </div>
      <div className="flex items-center justify-between text-xs">
        <span className="font-mono text-accent-cyan">{species.prob.toFixed(3)}</span>
        <div className="flex gap-1">
          {!isBird   && <span className="label-chip text-accent-amber">non-bird</span>}
          {species.well_trained && <span className="label-chip text-accent-lime">trained</span>}
        </div>
      </div>
    </Link>
  )
}
