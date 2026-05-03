import { useEffect, useState } from 'react'

interface Props {
  jobId: number
  onDone: (url: string) => void
  onError: (msg: string) => void
}

async function pollJob(jobId: number): Promise<{ status: string; output_path?: string }> {
  const res = await fetch(`/job/${jobId}`)
  if (!res.ok) throw new Error('Network error')
  return res.json()
}

export default function ProgressViewer({ jobId, onDone, onError }: Props) {
  const [stage, setStage] = useState('Starting…')

  useEffect(() => {
    let cancelled = false

    async function tick() {
      try {
        const data = await pollJob(jobId)
        if (cancelled) return

        if (data.status === 'running') {
          setStage('Pipeline running — this may take a few minutes')
        } else if (data.status === 'done' && data.output_path) {
          // output_path is stored as a relative path from the results dir,
          // e.g. "42/abc123_image.jpg" → served at /results/42/abc123_image.jpg
          const relPath = data.output_path.replace(/\\/g, '/')
          onDone(`/results/${relPath}`)
          return
        } else if (data.status === 'failed') {
          onError('Generation failed on the server.')
          return
        }
      } catch {
        if (!cancelled) onError('Lost connection to the backend.')
        return
      }

      if (!cancelled) {
        id = window.setTimeout(tick, 3000)
      }
    }

    // Fire immediately, then every 3 s
    let id = window.setTimeout(tick, 0)

    return () => {
      cancelled = true
      clearTimeout(id)
    }
  }, [jobId, onDone, onError])

  return (
    <div className="card progress-viewer">
      <div className="spinner" />
      <div className="progress-label">Generating…</div>
      <div className="progress-stage">{stage}</div>
    </div>
  )
}
