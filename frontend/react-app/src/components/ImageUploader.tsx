interface Props {
  contentPreview: string
  stylePreview: string
  onContentChange: (file: File) => void
  onStyleChange: (file: File) => void
}

export default function ImageUploader({ contentPreview, stylePreview, onContentChange, onStyleChange }: Props) {
  return (
    <div className="card">
      <div className="card-title">Images</div>
      <div className="upload-grid">
        <label className="dropzone">
          <input
            type="file"
            accept="image/*"
            onChange={e => e.target.files?.[0] && onContentChange(e.target.files[0])}
          />
          {contentPreview ? (
            <img src={contentPreview} alt="content preview" className="dropzone-preview" />
          ) : (
            <>
              <span className="dropzone-icon">🖼️</span>
              <span className="dropzone-label">Content</span>
              <span className="dropzone-sub">Click to upload</span>
            </>
          )}
        </label>

        <label className="dropzone">
          <input
            type="file"
            accept="image/*"
            onChange={e => e.target.files?.[0] && onStyleChange(e.target.files[0])}
          />
          {stylePreview ? (
            <img src={stylePreview} alt="style preview" className="dropzone-preview" />
          ) : (
            <>
              <span className="dropzone-icon">🎨</span>
              <span className="dropzone-label">Style</span>
              <span className="dropzone-sub">Optional</span>
            </>
          )}
        </label>
      </div>
    </div>
  )
}
