import { useState, useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Upload, Mic, FileAudio, Settings2, Zap, X,
  CheckCircle2, AlertTriangle, Sparkles, Music
} from 'lucide-react'
import Plot from 'react-plotly.js'
import { predict, getSpectrogram, getWaveform } from '../api/client'
import SpeciesCard from '../components/SpeciesCard'

export default function PredictPage() {
  const [audioFile, setAudioFile]         = useState(null)
  const [audioURL,  setAudioURL]          = useState(null)
  const [waveform,  setWaveform]          = useState(null)
  const [spectro,   setSpectro]           = useState(null)
  const [result,    setResult]            = useState(null)
  const [loading,   setLoading]           = useState(false)
  const [error,     setError]             = useState(null)

  // Settings
  const [topK,             setTopK]              = useState(3)
  const [noisePreprocess,  setNoisePreprocess]   = useState(true)
  const [onlyWellTrained,  setOnlyWellTrained]   = useState(true)
  const [consistencyBoost, setConsistencyBoost]  = useState(0.05)
  const [selectedChunk,    setSelectedChunk]     = useState(0)

  const onDrop = useCallback((files) => {
    const f = files[0]
    if (!f) return
    setAudioFile(f)
    setAudioURL(URL.createObjectURL(f))
    setResult(null)
    setWaveform(null)
    setSpectro(null)
    setError(null)
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'audio/*': ['.wav','.mp3','.ogg','.flac','.m4a'] },
    maxFiles: 1,
  })

  const runPrediction = async () => {
    if (!audioFile) return
    setLoading(true)
    setError(null)
    try {
      // Run waveform + prediction in parallel
      const [waveResp, predResp] = await Promise.all([
        getWaveform(audioFile, 1024),
        predict(audioFile, {
          topK,
          noisePreprocess,
          consistencyBoost,
          onlyWellTrained,
        }),
      ])
      setWaveform(waveResp)
      setResult(predResp)
      // Fetch spectrogram for chunk 0
      const specResp = await getSpectrogram(audioFile, 0, 5)
      setSpectro(specResp)
      setSelectedChunk(0)
    } catch (e) {
      setError(e.response?.data?.detail || e.message || 'Prediction failed')
    } finally {
      setLoading(false)
    }
  }

  const onSelectChunk = async (idx) => {
    setSelectedChunk(idx)
    if (audioFile) {
      try {
        const specResp = await getSpectrogram(audioFile, idx * 5, 5)
        setSpectro(specResp)
      } catch (e) {
        console.error(e)
      }
    }
  }

  const clear = () => {
    setAudioFile(null)
    setAudioURL(null)
    setWaveform(null)
    setSpectro(null)
    setResult(null)
    setError(null)
  }

  return (
    <div className="space-y-8">
      <div>
        <div className="heading-mono mb-2">// Predict</div>
        <h1 className="heading-display text-4xl">Per-5-second species prediction</h1>
        <p className="text-muted-light mt-2 max-w-2xl">
          Upload an audio clip. An AST (AudioSet) gate decides per 5-sec chunk
          whether it contains a bird — speech, music, traffic, silence, etc. are
          labeled as such. Bird chunks then pass to the 4-checkpoint species
          ensemble for per-window top-K classification.
        </p>
      </div>

      {/* ─── Upload + settings ──────────────────────────────────────── */}
      <div className="grid lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          {!audioFile ? (
            <div
              {...getRootProps()}
              className={`card-glow p-12 cursor-pointer text-center transition-all
                          ${isDragActive ? 'border-accent-cyan/60 bg-accent-cyan/5' : ''}`}
            >
              <input {...getInputProps()} />
              <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-accent-cyan/10 border border-accent-cyan/30 mb-4">
                <Upload className="w-8 h-8 text-accent-cyan" />
              </div>
              <h3 className="font-display text-xl font-semibold mb-2">
                {isDragActive ? 'Drop the audio here…' : 'Drop audio file here'}
              </h3>
              <p className="text-muted text-sm">
                or click to browse. Supports WAV, MP3, OGG, FLAC, M4A.
              </p>
              <div className="mt-4 text-xs text-muted-light font-mono">
                Recommended: 5-30 sec clip, ≥32 kHz, single bird, focal recording
              </div>
            </div>
          ) : (
            <div className="card p-6 space-y-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <FileAudio className="w-5 h-5 text-accent-cyan" />
                  <div>
                    <div className="font-medium">{audioFile.name}</div>
                    <div className="text-xs text-muted">
                      {(audioFile.size / 1024).toFixed(1)} KB
                    </div>
                  </div>
                </div>
                <button onClick={clear} className="btn-ghost">
                  <X className="w-4 h-4" />
                </button>
              </div>

              <audio src={audioURL} controls className="w-full" />

              <button
                onClick={runPrediction}
                disabled={loading}
                className="btn-primary w-full flex items-center justify-center gap-2 disabled:opacity-50"
              >
                {loading ? (
                  <><Sparkles className="w-4 h-4 animate-spin" /> Running ensemble inference…</>
                ) : (
                  <><Zap className="w-4 h-4" /> Run prediction</>
                )}
              </button>
              {loading && (
                <div className="text-xs text-muted-light text-center font-mono">
                  4-checkpoint ensemble + TTA on CPU — may take 2–4 min for &gt;10 s audio
                </div>
              )}
            </div>
          )}
        </div>

        {/* ─── Settings panel ──────────────────────────── */}
        <div className="card p-5 space-y-4">
          <div className="flex items-center gap-2 text-accent-cyan">
            <Settings2 className="w-4 h-4" />
            <div className="heading-mono">// Settings</div>
          </div>

          <Slider
            label="Top-K species" min={1} max={10} step={1}
            value={topK} onChange={setTopK}
          />
          <Slider
            label="Consistency boost" min={0} max={0.20} step={0.01}
            value={consistencyBoost} onChange={setConsistencyBoost}
            hint="Boost species seen in adjacent chunks"
          />
          <div className="text-[10px] text-muted-light leading-relaxed border-t border-ink-500/40 pt-3">
            <span className="text-accent-cyan font-mono">// non-bird detection</span><br />
            Handled automatically by AST (AudioSet) gate. Chunks of speech, music, traffic, silence etc. are labeled with the detected category instead of a species.
          </div>
          <Toggle
            label="Noise preprocessing"
            value={noisePreprocess} onChange={setNoisePreprocess}
            hint="Spectral gating + bandpass + pre-emphasis"
          />
          <Toggle
            label="Well-trained species only"
            value={onlyWellTrained} onChange={setOnlyWellTrained}
            hint="Hide rare/iNat numeric taxa"
          />
        </div>
      </div>

      {/* ─── Error ──────────────────────────────────────────────────── */}
      {error && (
        <div className="card border-accent-rose/40 bg-accent-rose/5 p-4 flex items-center gap-3">
          <AlertTriangle className="w-5 h-5 text-accent-rose" />
          <div className="text-sm">{error}</div>
        </div>
      )}

      {/* ─── Results ─────────────────────────────────────────────────── */}
      <AnimatePresence>
        {result && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4 }}
            className="space-y-6"
          >
            {/* Summary cards */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <SummaryCard label="Total chunks"    value={result.n_chunks} accent="cyan" />
              <SummaryCard label="Bird detected"   value={result.n_birds} accent="lime" />
              <SummaryCard label="No bird"          value={result.n_no_bird} accent="amber" />
              <SummaryCard label="Duration"        value={`${result.duration_sec}s`} accent="violet" />
            </div>

            {/* ─── PRIMARY: Per-5-second predictions ─────────────────── */}
            <div className="card p-6">
              <div className="flex items-center gap-2 mb-2">
                <Music className="w-5 h-5 text-accent-cyan" />
                <div className="heading-mono">// Per-5-second predictions</div>
              </div>
              <p className="text-xs text-muted-light mb-4">
                Top-{topK} species in each 5-second window. Click a window header to view its mel-spectrogram below.
              </p>
              <div className="grid md:grid-cols-2 gap-4">
                {result.chunks.map((c) => (
                  <ChunkPanel
                    key={c.chunk_idx}
                    chunk={c}
                    isSelected={selectedChunk === c.chunk_idx}
                    onSelect={() => onSelectChunk(c.chunk_idx)}
                  />
                ))}
              </div>
            </div>

            {/* Waveform */}
            {waveform && (
              <div className="card p-6">
                <div className="heading-mono mb-3">// Waveform</div>
                <Plot
                  data={[{
                    y: waveform.samples,
                    type: 'scatter',
                    mode: 'lines',
                    line: { color: '#00f0ff', width: 1 },
                    fill: 'tozeroy',
                    fillcolor: 'rgba(0,240,255,0.1)',
                    hoverinfo: 'skip',
                  }]}
                  layout={{
                    autosize: true,
                    height: 130,
                    margin: { l: 30, r: 10, t: 5, b: 25 },
                    paper_bgcolor: 'transparent',
                    plot_bgcolor:  'transparent',
                    xaxis: { showgrid: false, zeroline: false, color: '#7a8290', tickfont: { size: 9 } },
                    yaxis: { showgrid: false, zeroline: false, range: [-1.05, 1.05], showticklabels: false },
                    shapes: result.chunks?.map((c, i) => ({
                      type: 'rect',
                      xref: 'x', yref: 'paper',
                      x0: (i / result.n_chunks) * waveform.samples.length,
                      x1: ((i + 1) / result.n_chunks) * waveform.samples.length,
                      y0: 0, y1: 1,
                      line: { color: i === selectedChunk ? '#a3ff00' : 'transparent', width: 1.5 },
                      fillcolor: c.is_bird ? 'rgba(0,240,255,0.05)' : 'rgba(255,107,107,0.05)',
                    })),
                  }}
                  config={{ displayModeBar: false, responsive: true }}
                  style={{ width: '100%' }}
                />
              </div>
            )}

            {/* Spectrogram — full width, driven by selected chunk above */}
            <div className="card p-6">
              <div className="flex items-center justify-between mb-3">
                <div className="heading-mono">
                  // Mel-spectrogram · chunk {selectedChunk + 1} of {result.n_chunks}
                </div>
                <div className="text-xs text-muted">
                  {selectedChunk * 5}s – {(selectedChunk + 1) * 5}s
                </div>
              </div>
              {spectro && spectro.z && (
                <Plot
                  data={[{
                    z: spectro.z,
                    type: 'heatmap',
                    colorscale: 'Magma',
                    showscale: true,
                    colorbar: { title: 'dB', titlefont: { color: '#7a8290' }, tickfont: { color: '#7a8290' } },
                  }]}
                  layout={{
                    autosize: true,
                    height: 320,
                    margin: { l: 50, r: 10, t: 10, b: 40 },
                    paper_bgcolor: 'transparent',
                    plot_bgcolor:  'transparent',
                    xaxis: { title: 'time frame', color: '#7a8290', gridcolor: 'rgba(255,255,255,0.05)' },
                    yaxis: { title: 'mel bin',   color: '#7a8290', gridcolor: 'rgba(255,255,255,0.05)' },
                  }}
                  config={{ displayModeBar: false, responsive: true }}
                  style={{ width: '100%' }}
                />
              )}
            </div>

            {/* Aggregate (averaged) — moved to bottom, secondary */}
            {result.overall_top?.length > 0 && (
              <div className="card p-6">
                <div className="flex items-center gap-2 mb-1">
                  <Music className="w-4 h-4 text-accent-lime" />
                  <div className="heading-mono text-sm">// Aggregate across all chunks</div>
                </div>
                <p className="text-xs text-muted-light mb-4">
                  Mean probability over all {result.n_chunks} windows — recording-level summary, not a per-second prediction.
                </p>
                <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-3">
                  {result.overall_top.map((s) => (
                    <SpeciesCard key={s.code} species={s} variant="overall" />
                  ))}
                </div>
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

function ChunkPanel({ chunk, isSelected, onSelect }) {
  return (
    <div
      className={`card p-4 transition-all ${
        isSelected ? 'border-accent-cyan/60 bg-accent-cyan/5' : ''
      }`}
    >
      <button
        onClick={onSelect}
        className="w-full flex items-center justify-between mb-3 group"
      >
        <div className="flex items-center gap-2 font-mono text-xs">
          <span className="text-muted-light group-hover:text-accent-cyan transition-colors">
            chunk {chunk.chunk_idx + 1}
          </span>
          <span className="text-muted">·</span>
          <span className="text-muted-light">
            {chunk.start_sec}s – {chunk.end_sec}s
          </span>
          {isSelected && (
            <span className="text-accent-cyan text-[10px] ml-1">[spectrogram]</span>
          )}
        </div>
        {chunk.is_bird ? (
          <span className="label-chip text-accent-lime border-accent-lime/40 inline-flex items-center gap-1">
            <CheckCircle2 className="w-3 h-3" /> bird
          </span>
        ) : (
          <span className="label-chip text-accent-amber border-accent-amber/40 inline-flex items-center gap-1">
            <AlertTriangle className="w-3 h-3" /> non-bird audio
          </span>
        )}
      </button>

      {chunk.is_bird ? (
        <div className="space-y-1.5">
          {chunk.top_species.map((s, i) => (
            <SpeciesCard key={s.code} species={s} rank={i + 1} variant="row" />
          ))}
        </div>
      ) : (
        <div className="rounded-lg border border-accent-amber/30 bg-accent-amber/5 px-3 py-3">
          <div className="text-xs text-muted-light mb-1">Detected as:</div>
          <div className="font-display font-semibold text-accent-amber">
            {chunk.non_bird_label || 'Unknown non-bird sound'}
          </div>
          {chunk.non_bird_score != null && (
            <div className="font-mono text-[11px] text-muted mt-1">
              AST confidence {chunk.non_bird_score.toFixed(3)}
              <span className="text-muted-light"> · bird-likeness {chunk.gate_bird_score?.toFixed(3)}</span>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function SummaryCard({ label, value, accent }) {
  const colors = {
    cyan:   'text-accent-cyan',
    lime:   'text-accent-lime',
    amber:  'text-accent-amber',
    violet: 'text-accent-violet',
  }
  return (
    <div className="card-glow p-4">
      <div className="data-stat">{label}</div>
      <div className={`data-value ${colors[accent]}`}>{value}</div>
    </div>
  )
}

function Slider({ label, min, max, step, value, onChange, hint }) {
  return (
    <div>
      <div className="flex items-center justify-between mb-1">
        <label className="text-xs font-medium text-muted-light">{label}</label>
        <span className="font-mono text-xs text-accent-cyan">{value}</span>
      </div>
      <input
        type="range" min={min} max={max} step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full h-1.5 bg-ink-600 rounded-lg appearance-none accent-accent-cyan cursor-pointer"
      />
      {hint && <div className="text-[10px] text-muted mt-1">{hint}</div>}
    </div>
  )
}

function Toggle({ label, value, onChange, hint }) {
  return (
    <div>
      <label className="flex items-center justify-between cursor-pointer">
        <span className="text-xs font-medium text-muted-light">{label}</span>
        <button
          onClick={() => onChange(!value)}
          className={`w-9 h-5 rounded-full transition-colors relative
            ${value ? 'bg-accent-cyan' : 'bg-ink-500'}`}
        >
          <span className={`absolute top-0.5 left-0.5 w-4 h-4 bg-white rounded-full
                            transition-transform ${value ? 'translate-x-4' : ''}`} />
        </button>
      </label>
      {hint && <div className="text-[10px] text-muted mt-1">{hint}</div>}
    </div>
  )
}
