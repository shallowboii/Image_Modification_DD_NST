interface Props {
  resultUrl: string
  onReset: () => void
}

export default function ResultViewer({ resultUrl, onReset }: Props) {
  return (
    <div className="card result-viewer">
      <div className="card-title">Result</div>
      <img src={resultUrl} alt="Generated result" className="result-image" />
      <div className="result-actions">
        <a href={resultUrl} download className="result-btn primary">
          Download
        </a>
        <button className="result-btn" onClick={onReset}>
          New image
        </button>
      </div>
    </div>
  )
}
