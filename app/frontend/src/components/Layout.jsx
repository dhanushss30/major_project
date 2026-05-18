import { NavLink, Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import {
  Home, Mic, BarChart3, Search, Cpu, MessageSquare, Activity
} from 'lucide-react'

const NAV = [
  { to: '/',          label: 'Home',       icon: Home },
  { to: '/predict',   label: 'Predict',    icon: Mic },
  { to: '/dashboard', label: 'Dashboard',  icon: BarChart3 },
  { to: '/species',   label: 'Species',    icon: Search },
  { to: '/model',     label: 'Model',      icon: Cpu },
  { to: '/chat',      label: 'AI Chat',    icon: MessageSquare },
]

export default function Layout({ children }) {
  return (
    <div className="min-h-screen flex flex-col">
      {/* ── Top bar ──────────────────────────────────────────────────── */}
      <header className="sticky top-0 z-50 border-b border-ink-500/30 backdrop-blur-md bg-ink-900/70">
        <div className="container mx-auto px-6 py-3 flex items-center justify-between">
          <Link to="/" className="flex items-center gap-3 group">
            <div className="relative">
              <Activity className="w-7 h-7 text-accent-cyan group-hover:scale-110 transition-transform" />
              <div className="absolute inset-0 blur-md bg-accent-cyan/40 group-hover:bg-accent-cyan/60 transition-all" />
            </div>
            <div>
              <div className="font-display text-lg font-bold leading-none">
                Wild<span className="text-accent-cyan">Ear</span>
              </div>
              <div className="font-mono text-[10px] text-muted uppercase tracking-widest">
                v1.0 · macro-AUC 0.8246
              </div>
            </div>
          </Link>

          <nav className="hidden md:flex items-center gap-1">
            {NAV.map(({ to, label, icon: Icon }) => (
              <NavLink
                key={to}
                to={to}
                end={to === '/'}
                className={({ isActive }) =>
                  `flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium transition-all
                   ${isActive
                     ? 'text-accent-cyan bg-accent-cyan/10 border border-accent-cyan/30'
                     : 'text-muted-light hover:text-white hover:bg-ink-700'}`
                }
              >
                <Icon className="w-4 h-4" />
                {label}
              </NavLink>
            ))}
          </nav>

        </div>
      </header>

      {/* ── Main ─────────────────────────────────────────────────────── */}
      <main className="flex-1 container mx-auto px-6 py-8">
        <motion.div
          key={location.pathname}
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
        >
          {children}
        </motion.div>
      </main>

      {/* ── Footer ───────────────────────────────────────────────────── */}
      <footer className="border-t border-ink-500/30 py-6 mt-12">
        <div className="container mx-auto px-6 flex flex-col md:flex-row items-center justify-between gap-4 text-sm">
          <div className="text-muted font-mono text-xs">
            WildEar · ECA-NFNet-L0 ensemble · 206 species · 0.8246 macro ROC AUC
          </div>
          <div className="flex items-center gap-3 text-muted text-xs">
            <span className="label-chip">PyTorch</span>
            <span className="label-chip">FastAPI</span>
            <span className="label-chip">React + Vite</span>
            <span className="label-chip-accent">Groq RAG</span>
          </div>
        </div>
      </footer>
    </div>
  )
}
