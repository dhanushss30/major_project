import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import {
  Mic, BarChart3, Search, Cpu, MessageSquare,
  Sparkles, TrendingUp, Layers, Zap, ArrowRight, GitBranch
} from 'lucide-react'

const FEATURES = [
  {
    icon: Mic,
    title: 'Predict',
    desc: 'Upload or live-record audio. Get per-5-second species predictions with confidence bars and mel-spectrogram visualization.',
    to: '/predict',
    color: 'cyan',
  },
  {
    icon: BarChart3,
    title: 'Dashboard',
    desc: 'Real-time model metrics: per-class AUC histogram, training trajectory, ensemble vs single-checkpoint comparison.',
    to: '/dashboard',
    color: 'lime',
  },
  {
    icon: Search,
    title: 'Species Explorer',
    desc: 'Browse all 206 BirdCLEF classes. Filter by family, training-sample count, per-class AUC.',
    to: '/species',
    color: 'amber',
  },
  {
    icon: Cpu,
    title: 'Model Details',
    desc: 'Architecture diagram, training methodology, novel research findings, and complete ensemble breakdown.',
    to: '/model',
    color: 'violet',
  },
  {
    icon: MessageSquare,
    title: 'AI Chat',
    desc: 'Ask anything about the model methodology, individual species, or bird identification. Powered by Groq + RAG.',
    to: '/chat',
    color: 'rose',
  },
]

const STATS = [
  { label: 'Macro AUC',          value: '0.8246', accent: 'cyan' },
  { label: 'Classes',            value: '206',    accent: 'lime' },
  { label: 'Ensemble Ckpts',     value: '4',      accent: 'amber' },
  { label: 'Validation Files',   value: '5,710',  accent: 'violet' },
]

const COLORMAP = {
  cyan:   { bg: 'bg-accent-cyan/10',  border: 'border-accent-cyan/30',  text: 'text-accent-cyan'  },
  lime:   { bg: 'bg-accent-lime/10',  border: 'border-accent-lime/30',  text: 'text-accent-lime'  },
  amber:  { bg: 'bg-accent-amber/10', border: 'border-accent-amber/30', text: 'text-accent-amber' },
  violet: { bg: 'bg-accent-violet/10',border: 'border-accent-violet/30',text: 'text-accent-violet'},
  rose:   { bg: 'bg-accent-rose/10',  border: 'border-accent-rose/30',  text: 'text-accent-rose'  },
}

export default function HomePage() {
  return (
    <div className="space-y-16">

      {/* ─── Hero ────────────────────────────────────────────────────── */}
      <section className="relative">
        <div className="absolute inset-0 grid-lines opacity-30 -z-10 rounded-3xl" />
        <div className="relative max-w-4xl mx-auto text-center py-20">
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.5 }}
          >
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-accent-cyan/10 border border-accent-cyan/30 text-accent-cyan text-xs font-mono mb-6">
              <Sparkles className="w-3 h-3" />
              RESEARCH PROTOTYPE · v1.0
            </div>
            <h1 className="font-display text-5xl md:text-6xl font-bold leading-tight text-balance mb-6">
              Neotropical bird species classifier
              <span className="block mt-2 bg-gradient-to-r from-accent-cyan via-accent-lime to-accent-amber bg-clip-text text-transparent">
                WildEar
              </span>
            </h1>
            <p className="text-muted-light text-lg max-w-2xl mx-auto text-balance leading-relaxed">
              A 4-checkpoint ECA-NFNet-L0 ensemble with test-time augmentation,
              noise-robust preprocessing, and open-set rejection. Achieves
              <span className="font-mono text-accent-cyan glow-text-cyan"> 0.8246 macro ROC AUC </span>
              on a 206-species Neotropical bird validation set.
            </p>
            <div className="mt-8 flex flex-wrap items-center justify-center gap-3">
              <Link to="/predict" className="btn-primary flex items-center gap-2">
                <Mic className="w-4 h-4" />
                Try a prediction
                <ArrowRight className="w-4 h-4" />
              </Link>
              <Link to="/dashboard" className="btn-secondary flex items-center gap-2">
                <BarChart3 className="w-4 h-4" />
                View metrics
              </Link>
            </div>
          </motion.div>
        </div>
      </section>

      {/* ─── Stats ───────────────────────────────────────────────────── */}
      <section>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {STATS.map((s, i) => {
            const cmap = COLORMAP[s.accent]
            return (
              <motion.div
                key={s.label}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 + i * 0.05 }}
                className={`card-glow p-5 ${cmap.bg} ${cmap.border}`}
              >
                <div className="data-stat">{s.label}</div>
                <div className={`data-value mt-1 ${cmap.text}`}>{s.value}</div>
              </motion.div>
            )
          })}
        </div>
      </section>

      {/* ─── Features ────────────────────────────────────────────────── */}
      <section>
        <div className="mb-6">
          <div className="heading-mono mb-2">// Features</div>
          <h2 className="heading-display text-3xl">Explore the system</h2>
        </div>
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-5">
          {FEATURES.map(({ icon: Icon, title, desc, to, color }, i) => {
            const cmap = COLORMAP[color]
            return (
              <motion.div
                key={to}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.05 * i }}
              >
                <Link
                  to={to}
                  className={`card-glow block p-6 group h-full ${cmap.border} hover:border-opacity-60`}
                >
                  <div className={`inline-flex items-center justify-center w-12 h-12 rounded-lg ${cmap.bg} ${cmap.border} border mb-4 group-hover:scale-110 transition-transform`}>
                    <Icon className={`w-6 h-6 ${cmap.text}`} />
                  </div>
                  <h3 className="font-display text-xl font-semibold mb-2">{title}</h3>
                  <p className="text-muted-light text-sm leading-relaxed">{desc}</p>
                  <div className={`mt-4 inline-flex items-center gap-1 text-sm font-medium ${cmap.text}`}>
                    Explore <ArrowRight className="w-3 h-3 transition-transform group-hover:translate-x-1" />
                  </div>
                </Link>
              </motion.div>
            )
          })}
        </div>
      </section>

      {/* ─── Methodology highlights ──────────────────────────────────── */}
      <section>
        <div className="mb-6">
          <div className="heading-mono mb-2">// Architecture</div>
          <h2 className="heading-display text-3xl">Multi-checkpoint ensemble with TTA</h2>
        </div>
        <div className="card p-8">
          <div className="grid md:grid-cols-3 gap-6 mb-6">
            <FlowStep
              icon={Layers}
              n={1}
              title="Audio → mel-spec"
              desc="32 kHz mono, 5-sec chunks, 128-mel STFT"
            />
            <FlowStep
              icon={GitBranch}
              n={2}
              title="4-ckpt ensemble + TTA"
              desc="Averaged across architectures + overlapping windows"
            />
            <FlowStep
              icon={Zap}
              n={3}
              title="Post-process"
              desc="Noise gating · open-set rejection · consistency vote"
            />
          </div>
          <div className="grid md:grid-cols-2 gap-6 mt-6">
            <div>
              <div className="heading-mono mb-2">// Ensemble members</div>
              <ul className="space-y-2 text-sm">
                <CkptRow name="v3 fold 0 (clean)"  auc="0.7756" tag="clean" />
                <CkptRow name="v3 fold 0 (FiLM)"   auc="0.7841" tag="FiLM" />
                <CkptRow name="ESC-50 BG-trained"  auc="0.6814" tag="noise" />
                <CkptRow name="New fold 1 (clean)" auc="0.6976" tag="clean" />
                <li className="flex items-center justify-between font-mono text-sm pt-3 mt-3 border-t border-ink-500/40">
                  <span className="text-accent-cyan">Ensemble + TTA</span>
                  <span className="text-accent-cyan glow-text-cyan font-bold">0.8246</span>
                </li>
              </ul>
            </div>
            <div>
              <div className="heading-mono mb-2">// Novel research findings</div>
              <ul className="space-y-3 text-sm text-muted-light">
                <li className="flex gap-2">
                  <TrendingUp className="w-4 h-4 text-accent-lime flex-shrink-0 mt-0.5" />
                  <span>Rare-class pseudo-pollution diagnosis + masking algorithm</span>
                </li>
                <li className="flex gap-2">
                  <TrendingUp className="w-4 h-4 text-accent-lime flex-shrink-0 mt-0.5" />
                  <span>BG-augmentation labeling-contradiction analysis</span>
                </li>
                <li className="flex gap-2">
                  <TrendingUp className="w-4 h-4 text-accent-lime flex-shrink-0 mt-0.5" />
                  <span>Open-set rejection via max-prob thresholding</span>
                </li>
                <li className="flex gap-2">
                  <TrendingUp className="w-4 h-4 text-accent-lime flex-shrink-0 mt-0.5" />
                  <span>Inference-time noise-robust audio preprocessing pipeline</span>
                </li>
              </ul>
            </div>
          </div>
        </div>
      </section>
    </div>
  )
}

function FlowStep({ icon: Icon, n, title, desc }) {
  return (
    <div className="flex gap-4">
      <div className="flex flex-col items-center">
        <div className="w-10 h-10 rounded-full bg-accent-cyan/10 border border-accent-cyan/40 flex items-center justify-center font-mono text-sm text-accent-cyan">
          {n}
        </div>
      </div>
      <div>
        <div className="flex items-center gap-2 mb-1">
          <Icon className="w-4 h-4 text-accent-cyan" />
          <div className="font-semibold">{title}</div>
        </div>
        <div className="text-sm text-muted-light">{desc}</div>
      </div>
    </div>
  )
}

function CkptRow({ name, auc, tag }) {
  return (
    <li className="flex items-center justify-between font-mono text-sm">
      <span className="text-muted-light">{name}</span>
      <div className="flex items-center gap-2">
        <span className="label-chip">{tag}</span>
        <span className="text-white">{auc}</span>
      </div>
    </li>
  )
}
