import { useState, useCallback } from 'react'
import './App.css'
import ImageUploader from './components/ImageUploader'
import ControlPanel, { type CustomParams } from './components/ControlPanel'
import ProgressViewer from './components/ProgressViewer'
import ResultViewer from './components/ResultViewer'

type AppStatus = 'idle' | 'running' | 'done' | 'failed'

const DEFAULT_PARAMS: CustomParams = {
  cycles: '',
  lr: '',
  pyramid_size: '',
  pyramid_ratio: '',
  num_gradient_ascent_iterations: '',
  num_iterations_nst: '',
  style_weight: '',
  layer: '',
}

export default function App() {
  const [status, setStatus] = useState<AppStatus>('idle')
  const [modeTab, setModeTab] = useState<'preset' | 'custom'>('preset')
  const [contentFile, setContentFile] = useState<File | null>(null)
  const [styleFile, setStyleFile] = useState<File | null>(null)
  const [contentPreview, setContentPreview] = useState('')
  const [stylePreview, setStylePreview] = useState('')
  const [preset, setPreset] = useState('FAST')
  const [pipelineMode, setPipelineMode] = useState<'fixed' | 'adaptive'>('fixed')
  const [customParams, setCustomParams] = useState<CustomParams>(DEFAULT_PARAMS)
  const [jobId, setJobId] = useState<number | null>(null)
  const [resultUrl, setResultUrl] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  const handleContentChange = useCallback((file: File) => {
    setContentFile(file)
    setContentPreview(prev => {
      if (prev) URL.revokeObjectURL(prev)
      return URL.createObjectURL(file)
    })
  }, [])

  const handleStyleChange = useCallback((file: File) => {
    setStyleFile(file)
    setStylePreview(prev => {
      if (prev) URL.revokeObjectURL(prev)
      return URL.createObjectURL(file)
    })
  }, [])

  const handleCustomChange = useCallback((field: keyof CustomParams, value: string) => {
    setCustomParams(prev => ({ ...prev, [field]: value }))
  }, [])

  const handleGenerate = async () => {
    if (!contentFile) return
    setError(null)

    const form = new FormData()
    form.append('file', contentFile)
    if (styleFile) form.append('style', styleFile)

    if (modeTab === 'preset') {
      form.append('preset', preset)
    } else {
      for (const [key, val] of Object.entries(customParams)) {
        if (val !== '') form.append(key, val)
      }
    }
    // Always send pipeline mode regardless of which tab is active
    form.append('mode', pipelineMode)

    try {
      setStatus('running')
      const res = await fetch('/generate', { method: 'POST', body: form })
      if (!res.ok) throw new Error(`Server error ${res.status}`)
      const data: { job_id: number } = await res.json()
      setJobId(data.job_id)
    } catch (err) {
      setStatus('failed')
      setError(err instanceof Error ? err.message : 'Failed to start generation.')
    }
  }

  const handleDone = useCallback((url: string) => {
    setResultUrl(url)
    setStatus('done')
  }, [])

  const handleError = useCallback((msg: string) => {
    setError(msg)
    setStatus('failed')
  }, [])

  const handleReset = () => {
    setStatus('idle')
    setResultUrl(null)
    setJobId(null)
    setError(null)
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>Dream Studio</h1>
        <span>Deep Dream · Neural Style Transfer</span>
      </header>

      {error && <div className="error-banner">{error}</div>}

      <div className="app-body">
        <div>
          <ImageUploader
            contentPreview={contentPreview}
            stylePreview={stylePreview}
            onContentChange={handleContentChange}
            onStyleChange={handleStyleChange}
          />

          {status === 'done' && resultUrl && (
            <div style={{ marginTop: '1rem' }}>
              <ResultViewer resultUrl={resultUrl} onReset={handleReset} />
            </div>
          )}
        </div>

        <div>
          <ControlPanel
            modeTab={modeTab}
            onModeTabChange={setModeTab}
            preset={preset}
            onPresetChange={setPreset}
            pipelineMode={pipelineMode}
            onPipelineModeChange={setPipelineMode}
            customParams={customParams}
            onCustomChange={handleCustomChange}
            onGenerate={handleGenerate}
            disabled={!contentFile || status === 'running'}
          />

          {status === 'running' && jobId !== null && (
            <div style={{ marginTop: '1rem' }}>
              <ProgressViewer jobId={jobId} onDone={handleDone} onError={handleError} />
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
