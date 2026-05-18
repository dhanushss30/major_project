import { useEffect, useState } from 'react'
import { useParams, Link } from 'react-router-dom'
import ReactMarkdown from 'react-markdown'
import {
  Award, AlertCircle, Sparkles, ArrowLeft, Cpu, Tag, MapPin, Activity
} from 'lucide-react'
import { getSpeciesWithAI } from '../api/client'

export default function SpeciesDetailPage() {
  const { code } = useParams()
  const [info,    setInfo]    = useState(null)
  const [loading, setLoading] = useState(true)
  const [error,   setError]   = useState(null)

  useEffect(() => {
    setLoading(true); setError(null); setInfo(null)
    getSpeciesWithAI(code)
      .then(setInfo)
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [code])

  if (loading) return <Loading />
  if (error)   return <div className="card p-6 text-accent-rose">Error: {error}</div>
  if (!info)   return <div className="card p-6">Not found.</div>

  return (
    <div className="space-y-6">
      <Link to="/species" className="inline-flex items-center gap-1 text-sm text-muted hover:text-white">
        <ArrowLeft className="w-4 h-4" /> All species
      </Link>

      {/* ─── Header ─────────────────────────────────────────────────── */}
      <div className="card p-8">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <div className="font-mono text-sm text-accent-cyan mb-1">{info.code}</div>
            <h1 className="heading-display text-4xl">
              {info.common_name || <span className="text-muted">unnamed species</span>}
            </h1>
            {info.scientific_name && (
              <div className="font-display italic text-muted-light mt-1">{info.scientific_name}</div>
            )}
            <div className="flex gap-2 mt-3">
              {info.is_well_trained && (
                <span className="label-chip text-accent-lime border-accent-lime/40 flex items-center gap-1">
                  <Award className="w-3 h-3" /> Well-trained
                </span>
              )}
              {!info.is_bird && (
                <span className="label-chip text-accent-amber border-accent-amber/40 flex items-center gap-1">
                  <AlertCircle className="w-3 h-3" /> Non-bird taxon
                </span>
              )}
              {info.curated && (
                <span className="label-chip text-accent-cyan border-accent-cyan/40">Curated</span>
              )}
            </div>
          </div>
          <div className="grid grid-cols-2 gap-3 text-right">
            <Metric label="Train samples" value={info.n_train_samples} />
            <Metric label="val AUC" value={info.val_auc?.toFixed(3) ?? '—'}
                    accent={
                      info.val_auc > 0.9 ? 'lime'
                      : info.val_auc > 0.7 ? 'cyan'
                      : info.val_auc > 0.5 ? 'amber'
                      : 'rose'
                    } />
          </div>
        </div>
      </div>

      {/* ─── Curated info grid ───────────────────────────────────────── */}
      {info.curated && (
        <div className="grid md:grid-cols-2 gap-6">
          <div className="card p-6 space-y-3">
            <div className="heading-mono">// Taxonomy & description</div>
            {info.family && <Row icon={Tag}      label="Family"       value={info.family} />}
            {info.order  && <Row icon={Tag}      label="Order"        value={info.order} />}
            {info.habitat && <Row icon={MapPin}  label="Habitat"      value={info.habitat} />}
            {info.size_cm && <Row icon={Activity} label="Size"          value={`${info.size_cm} cm`} />}
            {info.conservation && <Row icon={Award} label="Conservation" value={info.conservation} />}
            {info.description && (
              <div className="pt-3 border-t border-ink-500/40">
                <div className="text-xs font-mono text-muted mb-2">Description</div>
                <p className="text-sm leading-relaxed text-muted-light">{info.description}</p>
              </div>
            )}
          </div>

          <div className="card p-6 space-y-3">
            <div className="heading-mono">// Model performance</div>
            <Row label="Train samples"  value={info.n_train_samples} icon={Cpu} />
            <Row label="Validation AUC" value={info.val_auc?.toFixed(4) ?? '—'} icon={Activity} />
            <Row label="Well-trained?"  value={info.is_well_trained ? 'Yes (≥50 samples)' : 'No'} icon={Award} />
            <Row label="Is bird?"        value={info.is_bird ? 'Yes' : 'No (iNat taxon)'} icon={AlertCircle} />
          </div>
        </div>
      )}

      {/* ─── AI-generated description ────────────────────────────────── */}
      {info.ai_description && (
        <div className="card p-6">
          <div className="flex items-center gap-2 mb-3">
            <Sparkles className="w-5 h-5 text-accent-violet" />
            <div className="heading-mono">// AI-generated context</div>
            <span className="label-chip text-accent-violet">Groq</span>
          </div>
          <div className="prose prose-invert prose-sm max-w-none text-muted-light leading-relaxed">
            <ReactMarkdown>{info.ai_description}</ReactMarkdown>
          </div>
        </div>
      )}
    </div>
  )
}

function Row({ icon: Icon, label, value }) {
  return (
    <div className="flex items-center gap-3 text-sm">
      {Icon && <Icon className="w-4 h-4 text-muted" />}
      <span className="text-muted w-32 text-xs font-mono">{label}</span>
      <span className="text-white">{value}</span>
    </div>
  )
}

function Metric({ label, value, accent = 'cyan' }) {
  const colors = {
    cyan: 'text-accent-cyan',
    lime: 'text-accent-lime',
    amber: 'text-accent-amber',
    rose: 'text-accent-rose',
  }
  return (
    <div>
      <div className="text-xs font-mono text-muted uppercase">{label}</div>
      <div className={`font-mono text-xl font-bold ${colors[accent]}`}>{value}</div>
    </div>
  )
}

function Loading() {
  return (
    <div className="space-y-4">
      <div className="card h-32 shimmer" />
      <div className="grid grid-cols-2 gap-4">
        <div className="card h-48 shimmer" />
        <div className="card h-48 shimmer" />
      </div>
    </div>
  )
}
