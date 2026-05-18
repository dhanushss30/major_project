import axios from 'axios'

const API_BASE = import.meta.env.VITE_API_BASE || '/api'

export const apiClient = axios.create({
  baseURL: API_BASE,
  timeout: 600_000,   // 10 min — CPU inference on long audio + 4 ckpts can take 2-4 min
})

// ── Predict ───────────────────────────────────────────────────────────────
export async function predict(audioFile, opts = {}) {
  const form = new FormData()
  form.append('audio', audioFile)
  form.append('top_k', opts.topK ?? 5)
  form.append('no_bird_thresh', opts.noBirdThresh ?? 0.10)
  form.append('tta_hop', opts.ttaHop ?? 2.5)
  form.append('noise_preprocess', opts.noisePreprocess ?? true)
  form.append('consistency_boost', opts.consistencyBoost ?? 0.05)
  form.append('only_well_trained', opts.onlyWellTrained ?? false)

  const { data } = await apiClient.post('/predict', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return data
}

// ── Audio visualization ───────────────────────────────────────────────────
export async function getWaveform(audioFile, nPoints = 1024) {
  const form = new FormData()
  form.append('audio', audioFile)
  form.append('n_points', nPoints)
  const { data } = await apiClient.post('/audio/waveform', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return data
}

export async function getSpectrogram(audioFile, startSec = 0, durationSec = 5) {
  const form = new FormData()
  form.append('audio', audioFile)
  form.append('start_sec', startSec)
  form.append('duration_sec', durationSec)
  const { data } = await apiClient.post('/audio/spectrogram', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return data
}

// ── Species ───────────────────────────────────────────────────────────────
export async function listSpecies(params = {}) {
  const { data } = await apiClient.get('/species/', { params })
  return data
}

export async function getSpecies(code, aiDescription = false) {
  const { data } = await apiClient.get(`/species/${code}`, {
    params: { ai_description: aiDescription },
  })
  return data
}

export async function getSpeciesWithAI(code) {
  const { data } = await apiClient.get(`/species/${code}/info`)
  return data
}

// ── Chat ──────────────────────────────────────────────────────────────────
export async function chatMessage(message, history = [], speciesCode = null) {
  const { data } = await apiClient.post('/chat/', {
    message,
    history,
    species_context: speciesCode,
  })
  return data
}

export async function chatStatus() {
  const { data } = await apiClient.get('/chat/status')
  return data
}

// ── Dashboard ─────────────────────────────────────────────────────────────
export async function getMetricsSummary() {
  const { data } = await apiClient.get('/metrics/summary')
  return data
}

export async function getPerClassAUC() {
  const { data } = await apiClient.get('/metrics/per-class-auc')
  return data
}

export async function getTrainingTrajectory() {
  const { data } = await apiClient.get('/metrics/training-trajectory')
  return data
}

// ── Root ──────────────────────────────────────────────────────────────────
export async function getServiceInfo() {
  const { data } = await axios.get('/')
  return data
}
