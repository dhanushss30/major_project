import { useState, useEffect, useRef } from 'react'
import ReactMarkdown from 'react-markdown'
import {
  Send, Sparkles, Bot, User, AlertCircle, Trash2
} from 'lucide-react'
import { chatMessage, chatStatus } from '../api/client'

const SAMPLE_PROMPTS = [
  "Explain the ensemble architecture in detail.",
  "What is the difference between val AUC and per-recording top-1 accuracy?",
  "Why does the model struggle with the Blue-grey Tanager (bugtan)?",
  "Tell me about the Tropical Kingbird (trokin).",
  "What is rare-class pseudo-pollution?",
  "How does open-set rejection work in this model?",
  "What can I do to improve this model in v2?",
]

export default function ChatPage() {
  const [enabled,  setEnabled]  = useState(null)
  const [messages, setMessages] = useState([])
  const [input,    setInput]    = useState('')
  const [loading,  setLoading]  = useState(false)
  const scrollRef = useRef(null)

  useEffect(() => {
    chatStatus().then(s => setEnabled(s.enabled)).catch(() => setEnabled(false))
  }, [])

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' })
  }, [messages, loading])

  const send = async (msg) => {
    const text = msg ?? input.trim()
    if (!text || loading) return
    const newMsgs = [...messages, { role: 'user', content: text }]
    setMessages(newMsgs)
    setInput('')
    setLoading(true)
    try {
      const history = newMsgs.slice(0, -1).map(m => ({ role: m.role, content: m.content }))
      const resp = await chatMessage(text, history)
      setMessages([...newMsgs, { role: 'assistant', content: resp.reply }])
    } catch (e) {
      setMessages([...newMsgs, { role: 'assistant', content: `Error: ${e.message}` }])
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-6 h-[calc(100vh-180px)] flex flex-col">
      <div className="flex items-start justify-between">
        <div>
          <div className="heading-mono mb-2">// AI Chat</div>
          <h1 className="heading-display text-4xl">Ask the model</h1>
          <p className="text-muted-light mt-2 max-w-2xl">
            RAG over the model documentation, species catalog, and
            research findings. Powered by Groq.
          </p>
        </div>
        <button
          onClick={() => setMessages([])}
          disabled={messages.length === 0}
          className="btn-ghost flex items-center gap-1 disabled:opacity-30"
        >
          <Trash2 className="w-4 h-4" /> Clear
        </button>
      </div>

      {enabled === false && (
        <div className="card border-accent-amber/30 bg-accent-amber/5 p-4 flex gap-3">
          <AlertCircle className="w-5 h-5 text-accent-amber flex-shrink-0" />
          <div className="text-sm text-muted-light">
            <div className="font-semibold text-accent-amber mb-1">AI chat disabled</div>
            Set <code className="font-mono text-xs px-1 py-0.5 bg-ink-700 rounded">GROQ_API_KEY</code> in
            <code className="font-mono text-xs px-1 py-0.5 bg-ink-700 rounded ml-1">backend/.env</code>
            {' '}and restart. Get a free key at <a href="https://console.groq.com/keys" target="_blank" rel="noreferrer"
            className="text-accent-cyan underline">console.groq.com/keys</a>.
          </div>
        </div>
      )}

      {/* ─── Chat area ─────────────────────────────────────────────── */}
      <div ref={scrollRef} className="card flex-1 overflow-y-auto p-6 space-y-4">
        {messages.length === 0 && (
          <div className="text-center py-12">
            <Bot className="w-12 h-12 mx-auto text-muted opacity-50 mb-4" />
            <div className="font-display text-lg text-muted mb-4">Try one of these:</div>
            <div className="grid sm:grid-cols-2 gap-2 max-w-2xl mx-auto">
              {SAMPLE_PROMPTS.map(p => (
                <button
                  key={p}
                  onClick={() => send(p)}
                  disabled={!enabled || loading}
                  className="text-left px-4 py-3 rounded-lg border border-ink-500/40 bg-ink-700/30 hover:border-accent-cyan/40 hover:bg-ink-700/60 transition-all text-sm text-muted-light hover:text-white disabled:opacity-40"
                >
                  {p}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((m, i) => (
          <div key={i} className={`flex gap-3 ${m.role === 'user' ? 'justify-end' : ''}`}>
            {m.role === 'assistant' && (
              <div className="w-8 h-8 rounded-full bg-accent-violet/10 border border-accent-violet/40 flex items-center justify-center flex-shrink-0">
                <Sparkles className="w-4 h-4 text-accent-violet" />
              </div>
            )}
            <div className={`max-w-[80%] rounded-2xl px-4 py-3 ${
              m.role === 'user'
                ? 'bg-accent-cyan/10 border border-accent-cyan/30 text-white'
                : 'bg-ink-700/60 border border-ink-500/40'
            }`}>
              {m.role === 'assistant' ? (
                <div className="prose prose-invert prose-sm max-w-none">
                  <ReactMarkdown>{m.content}</ReactMarkdown>
                </div>
              ) : (
                <div className="text-sm whitespace-pre-wrap">{m.content}</div>
              )}
            </div>
            {m.role === 'user' && (
              <div className="w-8 h-8 rounded-full bg-accent-cyan/10 border border-accent-cyan/40 flex items-center justify-center flex-shrink-0">
                <User className="w-4 h-4 text-accent-cyan" />
              </div>
            )}
          </div>
        ))}

        {loading && (
          <div className="flex gap-3">
            <div className="w-8 h-8 rounded-full bg-accent-violet/10 border border-accent-violet/40 flex items-center justify-center">
              <Sparkles className="w-4 h-4 text-accent-violet animate-pulse" />
            </div>
            <div className="bg-ink-700/60 border border-ink-500/40 rounded-2xl px-4 py-3">
              <div className="flex gap-1.5 items-center">
                <span className="w-1.5 h-1.5 rounded-full bg-accent-cyan animate-pulse" />
                <span className="w-1.5 h-1.5 rounded-full bg-accent-cyan animate-pulse" style={{ animationDelay: '0.2s' }} />
                <span className="w-1.5 h-1.5 rounded-full bg-accent-cyan animate-pulse" style={{ animationDelay: '0.4s' }} />
              </div>
            </div>
          </div>
        )}
      </div>

      {/* ─── Input ─────────────────────────────────────────────────── */}
      <form
        onSubmit={(e) => { e.preventDefault(); send() }}
        className="card flex items-center gap-3 p-3"
      >
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder={enabled ? "Ask anything about the model…" : "Set GROQ_API_KEY to enable chat"}
          disabled={!enabled || loading}
          className="flex-1 bg-transparent px-3 py-2 text-sm focus:outline-none disabled:opacity-40"
        />
        <button
          type="submit"
          disabled={!enabled || loading || !input.trim()}
          className="btn-primary flex items-center gap-1.5 disabled:opacity-30"
        >
          <Send className="w-4 h-4" /> Send
        </button>
      </form>
    </div>
  )
}
