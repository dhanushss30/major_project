import { Cpu, Layers, GitBranch, Zap, Shield, Microscope, AlertTriangle, ChevronRight } from 'lucide-react'

export default function ModelInfoPage() {
  return (
    <div className="space-y-8">
      <div>
        <div className="heading-mono mb-2">// Model</div>
        <h1 className="heading-display text-4xl">Architecture & methodology</h1>
        <p className="text-muted-light mt-2 max-w-3xl">
          Technical details of the 4-checkpoint ECA-NFNet-L0 ensemble, training
          methodology, novel research findings, and known limitations.
        </p>
      </div>

      {/* ─── Architecture ────────────────────────────────────────────── */}
      <section className="card p-8">
        <div className="flex items-center gap-2 mb-6">
          <Cpu className="w-5 h-5 text-accent-cyan" />
          <h2 className="heading-display text-2xl">Architecture</h2>
        </div>
        <div className="space-y-4">
          <KV k="Backbone"          v="ECA-NFNet-L0 (Efficient Channel Attention NFNet-L0, timm)" />
          <KV k="Parameters"        v="23.1M trainable" />
          <KV k="Input"             v="5-second mono audio @ 32 kHz → 128-mel STFT (n_fft=2048, hop=512)" />
          <KV k="Output"            v="206-class sigmoid probabilities (multi-label)" />
          <KV k="Loss"              v="BCE + Focal (0.5/0.5 weighted, focal γ=2.0) + label smoothing α=0.05" />
          <KV k="Optimizer"         v="RAdam, lr=5e-4, cosine schedule to 1e-6, batch=16×4 accumulation=64" />
          <KV k="CV strategy"       v="StratifiedGroupKFold (by species, grouped by recordist), 5 folds" />
          <KV k="Sampler"           v="Balanced sampler γ=−0.5 (SqrtBalancing), rating-weighted" />
          <KV k="EMA"               v="exponential moving average, decay 0.999" />
        </div>
      </section>

      {/* ─── Ensemble members ────────────────────────────────────────── */}
      <section className="card p-8">
        <div className="flex items-center gap-2 mb-6">
          <Layers className="w-5 h-5 text-accent-lime" />
          <h2 className="heading-display text-2xl">Ensemble members</h2>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-muted font-mono text-xs border-b border-ink-500/40">
                <th className="py-2">Name</th>
                <th>Architecture</th>
                <th>Training fold (val)</th>
                <th>val AUC</th>
                <th>Role</th>
              </tr>
            </thead>
            <tbody>
              <tr className="border-b border-ink-500/20">
                <td className="py-3 font-mono text-xs">v3 fold 0 (clean)</td>
                <td><span className="label-chip">clean</span></td>
                <td className="font-mono">fold 0</td>
                <td className="font-mono text-accent-cyan">0.7756</td>
                <td className="text-muted-light">Baseline reference</td>
              </tr>
              <tr className="border-b border-ink-500/20">
                <td className="py-3 font-mono text-xs">v3 fold 0 (FiLM)</td>
                <td><span className="label-chip text-accent-violet border-accent-violet/40">FiLM</span></td>
                <td className="font-mono">fold 0</td>
                <td className="font-mono text-accent-cyan">0.7841</td>
                <td className="text-muted-light">Architectural diversity</td>
              </tr>
              <tr className="border-b border-ink-500/20">
                <td className="py-3 font-mono text-xs">ESC-50 BG peak</td>
                <td><span className="label-chip text-accent-amber border-accent-amber/40">noise-aug</span></td>
                <td className="font-mono">fold 1</td>
                <td className="font-mono text-accent-amber">0.6814</td>
                <td className="text-muted-light">Noise robustness specialist</td>
              </tr>
              <tr>
                <td className="py-3 font-mono text-xs">New fold 1 (clean)</td>
                <td><span className="label-chip">clean</span></td>
                <td className="font-mono">fold 1</td>
                <td className="font-mono text-accent-cyan">0.6976</td>
                <td className="text-muted-light">Data-split diversity</td>
              </tr>
              <tr className="border-t-2 border-accent-cyan/30">
                <td className="py-3 font-mono text-xs font-bold text-accent-cyan">Ensemble + TTA</td>
                <td colSpan={2} className="text-muted">averaged sigmoid probs across all 4 ckpts + overlapping TTA windows</td>
                <td className="font-mono text-accent-cyan glow-text-cyan font-bold">0.8246</td>
                <td className="text-accent-cyan font-medium">Final result</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      {/* ─── Inference pipeline ──────────────────────────────────────── */}
      <section className="card p-8">
        <div className="flex items-center gap-2 mb-6">
          <Zap className="w-5 h-5 text-accent-amber" />
          <h2 className="heading-display text-2xl">Inference pipeline</h2>
        </div>
        <div className="space-y-3">
          {[
            { n: 1, t: 'Resample audio to 32 kHz mono' },
            { n: 2, t: 'Noise-robust preprocessing: spectral noise gating (subtract estimated noise floor) → bandpass 1–8 kHz → pre-emphasis filter' },
            { n: 3, t: 'Chunk into overlapping 5-second windows (TTA hop = 2.5s by default)' },
            { n: 4, t: 'Extract mel-spectrogram per chunk (128 mel bins, log-scaled)' },
            { n: 5, t: 'Forward pass through all 4 checkpoints in parallel, get sigmoid probabilities' },
            { n: 6, t: 'Average predictions across ckpts AND across TTA windows within each chunk' },
            { n: 7, t: 'Apply consistency vote: boost classes appearing in adjacent chunks\' top-K' },
            { n: 8, t: 'Open-set rejection: if max-prob < threshold → mark chunk as "no bird"' },
            { n: 9, t: 'Return top-K species per chunk with confidence scores' },
          ].map(s => (
            <div key={s.n} className="flex items-start gap-4 group">
              <div className="w-7 h-7 rounded-full bg-accent-cyan/10 border border-accent-cyan/30 flex items-center justify-center font-mono text-xs text-accent-cyan flex-shrink-0">
                {s.n}
              </div>
              <div className="text-muted-light text-sm leading-relaxed pt-1">{s.t}</div>
            </div>
          ))}
        </div>
      </section>

      {/* ─── Novel findings ──────────────────────────────────────────── */}
      <section className="card p-8">
        <div className="flex items-center gap-2 mb-6">
          <Microscope className="w-5 h-5 text-accent-violet" />
          <h2 className="heading-display text-2xl">Novel research findings</h2>
        </div>
        <div className="space-y-5">
          <Finding
            n={1}
            title="Rare-class pseudo-pollution in long-tail noisy-student"
            body="When teacher-student pseudo-labeling is applied to long-tail multi-label datasets, rare classes (78 of 206 taxa with <50 samples — mostly iNaturalist numeric IDs) get over-confidently memorized by the teacher. During pseudo-label generation, these rare classes dominate top-1 predictions on out-of-distribution soundscape audio, completely overwhelming the actual target species. Naive noisy-student iteration thus regresses model performance. We propose a post-hoc rare-class masking algorithm that zeros these columns in the pseudo-label parquet and recomputes max-confidence."
          />
          <Finding
            n={2}
            title="BG-augmentation labeling contradiction"
            body="Adding same-region soundscape audio as background noise during training (a common augmentation strategy) causes training collapse, because the soundscape backgrounds contain unlabeled target species. This creates a labeling contradiction: the model is asked to predict species absent (focal label) while their calls are physically present (in the BG track). Validation AUC drops from 0.7756 to 0.70. Mitigation: use out-of-distribution noise sources like ESC-50 (environmental sounds without birds), which preserves AUC while improving real-world noise robustness."
          />
          <Finding
            n={3}
            title="Open-set rejection via max-prob thresholding"
            body="For deployment, we extend the standard 206-class classifier with a 'no bird' verdict by thresholding the maximum sigmoid probability across all classes. The threshold (0.10 by default for this model) is calibrated to the model's conservative probability distribution (a side-effect of BCE + label smoothing). This allows the system to handle out-of-distribution audio (silence, music, human voice) without forcing a misleading bird prediction."
          />
          <Finding
            n={4}
            title="Inference-time noise robustness pipeline"
            body="Rather than achieve noise robustness through training augmentation (which we found can hurt val AUC), we add it at inference time: (a) spectral noise gating estimates the per-bin noise floor from quiet frames and subtracts it, (b) 1–8 kHz bandpass filter focuses on the bird-call frequency range, (c) pre-emphasis filter boosts high frequencies, (d) high-overlap TTA averages out chunk-specific noise. Provides ~+10-15% accuracy gain on noisy demo audio without retraining."
          />
        </div>
      </section>

      {/* ─── Limitations ─────────────────────────────────────────────── */}
      <section className="card p-8 border-accent-amber/30 bg-accent-amber/5">
        <div className="flex items-center gap-2 mb-6">
          <AlertTriangle className="w-5 h-5 text-accent-amber" />
          <h2 className="heading-display text-2xl">Known limitations</h2>
        </div>
        <ul className="space-y-3 text-sm text-muted-light">
          <Limit>Trained on focal (single-bird, clean) recordings — soundscape generalization is lower than the val AUC suggests</Limit>
          <Limit>78 of 206 classes have &lt;50 training samples and are unreliable; the model may predict these incorrectly on out-of-distribution audio</Limit>
          <Limit>Conservative probability calibration (max ~0.20 even for correct predictions; artifact of BCE + label smoothing) — ranking is reliable, absolute confidence is not</Limit>
          <Limit>Single-label training assumes one focal species per chunk; performance degrades on multi-species recordings</Limit>
          <Limit>Per-recording top-1 accuracy (~31% on tested clips) is significantly lower than the average macro AUC (0.82) — AUC measures ranking across many samples, not per-recording correctness</Limit>
          <Limit>FiLM noise-conditioning ckpt has known gamma-collapse issue producing near-constant outputs on OOD audio; included in ensemble for diversity but contributes flat predictions</Limit>
          <Limit>No "non-bird" class — open-set rejection is heuristic via max-prob threshold, not a learned discriminator</Limit>
        </ul>
      </section>

      {/* ─── Future work ─────────────────────────────────────────────── */}
      <section className="card p-8">
        <div className="flex items-center gap-2 mb-6">
          <ChevronRight className="w-5 h-5 text-accent-lime" />
          <h2 className="heading-display text-2xl">Future work (v2 roadmap)</h2>
        </div>
        <ul className="space-y-2 text-sm text-muted-light">
          <li>• Re-train with explicit negative samples (<code>p_negative=0.05</code>) for proper "no bird" detection</li>
          <li>• Train remaining folds 2, 3, 4 for full 5-fold ensemble (+0.02–0.04 expected AUC gain)</li>
          <li>• Try properly-fixed pseudo-labeling: mask zero-class positions in BCE to avoid the rare-class pollution found here</li>
          <li>• Add BirdNET-Lite as an ensemble member for production-grade real-world performance</li>
          <li>• Re-attempt failed novel modules with stability fixes: scheduled DANN, regularized FiLM, longer causal warmup</li>
          <li>• Multi-task learning: add region/season/habitat heads to force richer representations</li>
        </ul>
      </section>
    </div>
  )
}

function KV({ k, v }) {
  return (
    <div className="grid grid-cols-3 gap-4 py-2 border-b border-ink-500/20 last:border-0">
      <div className="font-mono text-xs text-muted uppercase tracking-wider">{k}</div>
      <div className="col-span-2 text-sm text-muted-light">{v}</div>
    </div>
  )
}

function Finding({ n, title, body }) {
  return (
    <div className="border-l-2 border-accent-violet/40 pl-4">
      <div className="flex items-center gap-2 mb-1">
        <span className="font-mono text-xs text-accent-violet">#{n}</span>
        <h3 className="font-semibold">{title}</h3>
      </div>
      <p className="text-sm text-muted-light leading-relaxed">{body}</p>
    </div>
  )
}

function Limit({ children }) {
  return (
    <li className="flex gap-2">
      <span className="text-accent-amber font-mono mt-1">·</span>
      <span>{children}</span>
    </li>
  )
}
