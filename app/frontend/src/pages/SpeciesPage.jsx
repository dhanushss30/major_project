import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { Search, Filter, Award, AlertCircle, ChevronRight } from 'lucide-react'
import { listSpecies } from '../api/client'

export default function SpeciesPage() {
  const [species,           setSpecies]        = useState([])
  const [filtered,          setFiltered]       = useState([])
  const [loading,           setLoading]        = useState(true)
  const [search,            setSearch]         = useState('')
  const [onlyBirds,         setOnlyBirds]      = useState(true)
  const [onlyWellTrained,   setOnlyWellTrained]= useState(false)

  useEffect(() => {
    listSpecies({}).then(d => {
      setSpecies(d.species)
      setLoading(false)
    }).catch(console.error)
  }, [])

  useEffect(() => {
    let f = species
    if (onlyBirds)        f = f.filter(s => s.is_bird)
    if (onlyWellTrained)  f = f.filter(s => s.is_well_trained)
    if (search) {
      const q = search.toLowerCase()
      f = f.filter(s =>
        s.code.toLowerCase().includes(q) ||
        (s.common_name || '').toLowerCase().includes(q) ||
        (s.scientific_name || '').toLowerCase().includes(q)
      )
    }
    setFiltered(f)
  }, [species, search, onlyBirds, onlyWellTrained])

  if (loading) {
    return (
      <div className="space-y-4">
        <div className="card h-12 shimmer" />
        {[...Array(8)].map((_,i) => <div key={i} className="card h-16 shimmer" />)}
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <div>
        <div className="heading-mono mb-2">// Species</div>
        <h1 className="heading-display text-4xl">All 206 classes</h1>
        <p className="text-muted-light mt-2">
          Browse all 206 species. Click any row for full details and AI-generated description.
        </p>
      </div>

      {/* ─── Filters ─────────────────────────────────────────────────── */}
      <div className="card p-4 flex flex-wrap items-center gap-3">
        <div className="flex-1 min-w-[220px] relative">
          <Search className="absolute left-3 top-2.5 w-4 h-4 text-muted" />
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search by code, common name, or scientific name…"
            className="w-full bg-ink-800 border border-ink-500/40 rounded-lg pl-10 pr-3 py-2 text-sm
                       placeholder-muted focus:outline-none focus:border-accent-cyan/40"
          />
        </div>
        <Filter className="w-4 h-4 text-muted" />
        <FilterToggle label="Birds only"        value={onlyBirds}        onChange={setOnlyBirds} />
        <FilterToggle label="Well-trained only" value={onlyWellTrained}  onChange={setOnlyWellTrained} />
        <div className="font-mono text-xs text-muted ml-auto">
          {filtered.length} / {species.length}
        </div>
      </div>

      {/* ─── List ─────────────────────────────────────────────────────── */}
      <div className="card overflow-hidden">
        <div className="grid grid-cols-12 px-4 py-2 border-b border-ink-500/30 text-xs font-mono text-muted uppercase tracking-wider">
          <div className="col-span-1">Code</div>
          <div className="col-span-4">Common name</div>
          <div className="col-span-3 hidden md:block">Scientific / Family</div>
          <div className="col-span-2 text-right">Train samples</div>
          <div className="col-span-2 text-right">val AUC</div>
        </div>
        <div className="divide-y divide-ink-500/20 max-h-[600px] overflow-y-auto">
          {filtered.slice(0, 250).map(s => (
            <Link
              key={s.code}
              to={`/species/${s.code}`}
              className="grid grid-cols-12 px-4 py-3 hover:bg-ink-700/40 transition-colors items-center"
            >
              <div className="col-span-1 font-mono text-xs text-accent-cyan">
                {s.code}
              </div>
              <div className="col-span-4 flex items-center gap-2 min-w-0">
                <span className="truncate">
                  {s.common_name || <span className="text-muted">— no name —</span>}
                </span>
                {s.is_well_trained && <Award className="w-3 h-3 text-accent-lime flex-shrink-0" />}
                {!s.is_bird && <AlertCircle className="w-3 h-3 text-accent-amber flex-shrink-0" />}
              </div>
              <div className="col-span-3 hidden md:block text-xs text-muted truncate">
                {s.scientific_name && <span className="italic">{s.scientific_name}</span>}
                {s.family && <span className="ml-2">· {s.family}</span>}
              </div>
              <div className="col-span-2 text-right font-mono text-sm">
                {s.n_train_samples}
              </div>
              <div className="col-span-2 text-right font-mono text-sm">
                {s.val_auc != null ? (
                  <span className={
                    s.val_auc > 0.9 ? 'text-accent-lime'
                    : s.val_auc > 0.7 ? 'text-accent-cyan'
                    : s.val_auc > 0.5 ? 'text-accent-amber'
                    : 'text-accent-rose'
                  }>{s.val_auc.toFixed(3)}</span>
                ) : (
                  <span className="text-muted">—</span>
                )}
              </div>
            </Link>
          ))}
        </div>
        {filtered.length > 250 && (
          <div className="px-4 py-3 text-center text-xs text-muted border-t border-ink-500/30">
            Showing first 250 of {filtered.length}. Refine search to see more.
          </div>
        )}
      </div>
    </div>
  )
}

function FilterToggle({ label, value, onChange }) {
  return (
    <button
      onClick={() => onChange(!value)}
      className={`px-3 py-1.5 rounded-lg text-xs font-medium border transition-all
        ${value
          ? 'border-accent-cyan/40 bg-accent-cyan/10 text-accent-cyan'
          : 'border-ink-500/40 text-muted-light hover:border-ink-400/60'}`}
    >
      {label}
    </button>
  )
}
