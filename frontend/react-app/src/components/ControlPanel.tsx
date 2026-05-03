export interface CustomParams {
  cycles: string
  lr: string
  pyramid_size: string
  pyramid_ratio: string
  num_gradient_ascent_iterations: string
  num_iterations_nst: string
  style_weight: string
  layer: string
}

const PRESETS = [
  { id: 'GENTLE',   icon: '🌿', name: 'Gentle',   desc: '1 cycle · subtle, preserves original' },
  { id: 'FAST',     icon: '⚡', name: 'Fast',     desc: '2 cycles · quick preview' },
  { id: 'Default',  icon: '✦',  name: 'Default',  desc: '2 cycles · balanced quality' },
  { id: 'PORTRAIT', icon: '🖼', name: 'Portrait',  desc: '2 cycles · soft, detail-preserving' },
  { id: 'LSD',      icon: '🌀', name: 'LSD',      desc: '3 cycles · vivid & surreal' },
  { id: 'DEEP',     icon: '🔮', name: 'Deep',     desc: '3 cycles · abstract, max depth' },
  { id: 'PAINTERLY', icon: '🎨', name: 'Painterly', desc: '1 cycle DD · 400-step NST dominant' },
  { id: 'HEAVY',    icon: '💀', name: 'Heavy',    desc: '4 cycles · maximum intensity' },
]

interface Props {
  modeTab: 'preset' | 'custom'
  onModeTabChange: (t: 'preset' | 'custom') => void
  preset: string
  onPresetChange: (p: string) => void
  pipelineMode: 'fixed' | 'adaptive'
  onPipelineModeChange: (m: 'fixed' | 'adaptive') => void
  customParams: CustomParams
  onCustomChange: (field: keyof CustomParams, value: string) => void
  onGenerate: () => void
  disabled: boolean
}

export default function ControlPanel({
  modeTab, onModeTabChange,
  preset, onPresetChange,
  pipelineMode, onPipelineModeChange,
  customParams, onCustomChange,
  onGenerate, disabled,
}: Props) {
  return (
    <div className="card">
      <div className="card-title">Settings</div>

      <div className="tabs">
        <button className={`tab-btn ${modeTab === 'preset' ? 'active' : ''}`} onClick={() => onModeTabChange('preset')}>
          Preset
        </button>
        <button className={`tab-btn ${modeTab === 'custom' ? 'active' : ''}`} onClick={() => onModeTabChange('custom')}>
          Custom
        </button>
      </div>

      {modeTab === 'preset' ? (
        <div className="preset-list">
          {PRESETS.map(p => (
            <button
              key={p.id}
              className={`preset-btn ${preset === p.id ? 'selected' : ''}`}
              onClick={() => onPresetChange(p.id)}
            >
              <span className="preset-icon">{p.icon}</span>
              <div>
                <div className="preset-name">{p.name}</div>
                <div className="preset-desc">{p.desc}</div>
              </div>
            </button>
          ))}
        </div>
      ) : (
        <div className="params-grid">
          {([
            ['cycles',                          'Cycles',      'number'],
            ['lr',                              'Dream LR',    'text'],
            ['pyramid_size',                    'Pyramid',     'number'],
            ['pyramid_ratio',                   'Pyr. ratio',  'text'],
            ['num_gradient_ascent_iterations',  'DD iters',    'number'],
            ['num_iterations_nst',              'NST steps',   'number'],
            ['style_weight',                    'Style w.',    'text'],
            ['layer',                           'DD layer',    'text'],
          ] as [keyof CustomParams, string, string][]).map(([key, label, type]) => (
            <div className="param-field" key={key}>
              <label htmlFor={key}>{label}</label>
              <input
                id={key}
                type={type}
                value={customParams[key]}
                placeholder="auto"
                onChange={e => onCustomChange(key, e.target.value)}
              />
            </div>
          ))}
        </div>
      )}

      {/* Pipeline mode toggle — shown in both tabs */}
      <div className="mode-toggle">
        {(['fixed', 'adaptive'] as const).map(m => (
          <button
            key={m}
            className={`mode-toggle-btn ${pipelineMode === m ? 'active' : ''}`}
            onClick={() => onPipelineModeChange(m)}
          >
            {m.charAt(0).toUpperCase() + m.slice(1)}
          </button>
        ))}
      </div>

      <button className="generate-btn" onClick={onGenerate} disabled={disabled}>
        {disabled ? 'Generating…' : 'Generate'}
      </button>
    </div>
  )
}
