import { useState } from 'react'
import axios from 'axios'
import './App.css'

function App() {
  const [selectedFile, setSelectedFile] = useState(null)
  const [playerSide, setPlayerSide] = useState('near')
  const [analysisMode, setAnalysisMode] = useState('match')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [activeSetTab, setActiveSetTab] = useState('all') // 'all' or set_number

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedFile(e.target.files[0])
      setResult(null)
      setError('')
    }
  }

  const handleUpload = async () => {
    if (!selectedFile) return
    setLoading(true)
    setError('')
    
    const formData = new FormData()
    formData.append('file', selectedFile)
    formData.append('player_side', playerSide)
    formData.append('analysis_mode', analysisMode)

    const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'
    try {
      const response = await axios.post(`${API_BASE_URL}/upload`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      })
      setResult(response.data)
      setActiveSetTab('all')
    } catch (err) {
      console.error(err)
      setError(
        err.response?.data?.detail ||
        "Échec de l'analyse. Vérifiez que votre backend est démarré ou l'URL configurée."
      )
    } finally {
      setLoading(false)
    }
  }

  // Rendu du tableau des statistiques cibles
  const renderStrokesTable = (strokesList) => {
    if (!strokesList || strokesList.length === 0) return null

    const totalStrokes = strokesList.reduce((acc, curr) => acc + curr.total, 0)
    const totalSuccessful = strokesList.reduce((acc, curr) => acc + curr.successful, 0)
    const totalErrors = strokesList.reduce((acc, curr) => acc + curr.unforced_errors, 0)
    const totalWinners = strokesList.reduce((acc, curr) => acc + curr.winners, 0)
    const overallSuccessRate = totalStrokes > 0 ? ((totalSuccessful / totalStrokes) * 100).toFixed(1) : 0

    const getStrokeIcon = (name) => {
      switch (name.toLowerCase()) {
        case 'topspin':
          return '🏓'
        case 'bloc':
          return '🛡️'
        case 'poussette':
          return '🪓'
        default:
          return '🎾'
      }
    }

    return (
      <div className="table-wrapper">
        <table className="stats-table">
          <thead>
            <tr>
              <th>Coup Technique</th>
              <th>Total Joués</th>
              <th>Réussis (%)</th>
              <th>Fautes Directes</th>
              <th>Points Gagnants (Winners)</th>
            </tr>
          </thead>
          <tbody>
            {strokesList.map((s) => (
              <tr key={s.stroke}>
                <td className="stroke-name">
                  <span className="stroke-icon">{getStrokeIcon(s.stroke)}</span>
                  <strong>{s.stroke}</strong>
                </td>
                <td className="metric-total">{s.total}</td>
                <td>
                  <div className="progress-cell">
                    <span className="success-badge">{s.successful} ({s.success_rate}%)</span>
                    <div className="mini-progress-bar">
                      <div className="bar-fill success-fill" style={{ width: `${s.success_rate}%` }}></div>
                    </div>
                  </div>
                </td>
                <td>
                  <span className="error-badge">{s.unforced_errors} ({s.error_rate}%)</span>
                </td>
                <td>
                  <div className="winner-cell">
                    <span className="winner-badge">🏆 {s.winners}</span>
                    <span className="winner-sub">({s.winner_rate}%)</span>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
          <tfoot>
            <tr className="total-row">
              <td><strong>{activeSetTab === 'all' ? 'TOTAL DU MATCH' : `TOTAL DU SET ${activeSetTab}`}</strong></td>
              <td><strong>{totalStrokes}</strong></td>
              <td>
                <span className="success-badge"><strong>{totalSuccessful} ({overallSuccessRate}%)</strong></span>
              </td>
              <td>
                <span className="error-badge"><strong>{totalErrors}</strong></span>
              </td>
              <td>
                <span className="winner-badge"><strong>🏆 {totalWinners}</strong></span>
              </td>
            </tr>
          </tfoot>
        </table>
      </div>
    )
  }

  const renderDashboard = () => {
    if (!result || result.mode !== 'match') return null
    const { summary, strokes_breakdown, sets, rallies } = result

    // Filtrage des coups selon l'onglet actif (Tous ou Set spécifique)
    let displayStrokes = strokes_breakdown
    let currentSetInfo = null
    if (activeSetTab !== 'all') {
      const selectedSet = sets?.find((s) => s.set_number === Number(activeSetTab))
      if (selectedSet) {
        displayStrokes = selectedSet.strokes
        currentSetInfo = selectedSet
      }
    }

    return (
      <section className="dashboard-section" id="results">
        {/* Titre & En-tête de résultats */}
        <div className="dashboard-header">
          <div>
            <h2 className="dashboard-title">Bilan de Performance du Match</h2>
            <p className="dashboard-subtitle">
              Vidéo analysée : <strong>{result.filename || 'Match complet'}</strong> • Durée : {summary.video_duration_sec}s
            </p>
          </div>
          <div className="match-status-badge">
            {summary.win_rate >= 50 ? '🎉 Match Positif' : '📈 Match Serré'}
          </div>
        </div>

        {/* 4 Cartes résumées principales */}
        <div className="summary-cards-grid">
          <div className="summary-card glass-panel">
            <span className="card-label">Points Gagnés</span>
            <div className="card-value-group">
              <span className="card-main-value text-accent">{summary.points_won}</span>
              <span className="card-sub-value">/ {summary.total_points} ({summary.win_rate}%)</span>
            </div>
            <div className="card-progress">
              <div className="progress-fill" style={{ width: `${summary.win_rate}%` }}></div>
            </div>
          </div>

          <div className="summary-card glass-panel">
            <span className="card-label">Taux de Réussite Global</span>
            <div className="card-value-group">
              <span className="card-main-value text-success">{summary.overall_success_rate}%</span>
              <span className="card-sub-value">{summary.total_successful} réussis</span>
            </div>
            <div className="card-progress">
              <div className="progress-fill fill-success" style={{ width: `${summary.overall_success_rate}%` }}></div>
            </div>
          </div>

          <div className="summary-card glass-panel">
            <span className="card-label">Points Gagnants (Winners)</span>
            <div className="card-value-group">
              <span className="card-main-value text-winner">🏆 {summary.total_winners}</span>
              <span className="card-sub-value">coups décisifs</span>
            </div>
            <p className="card-caption">Points achevés directement sur vos frappes</p>
          </div>

          <div className="summary-card glass-panel">
            <span className="card-label">Fautes Directes</span>
            <div className="card-value-group">
              <span className="card-main-value text-error">❌ {summary.total_unforced_errors}</span>
              <span className="card-sub-value">fautes</span>
            </div>
            <p className="card-caption">Balles données sur vos initiatives</p>
          </div>
        </div>

        {/* Onglets de filtrage par Set */}
        {sets && sets.length > 0 && (
          <div className="sets-navigation glass-panel">
            <div className="tabs-header">
              <span className="tabs-title">Décomposition par manche :</span>
              <div className="tabs-buttons">
                <button
                  type="button"
                  className={`set-tab-btn ${activeSetTab === 'all' ? 'active' : ''}`}
                  onClick={() => setActiveSetTab('all')}
                >
                  Vue Globale Match
                </button>
                {sets.map((set) => (
                  <button
                    key={set.set_number}
                    type="button"
                    className={`set-tab-btn ${activeSetTab === String(set.set_number) ? 'active' : ''}`}
                    onClick={() => setActiveSetTab(String(set.set_number))}
                  >
                    Set {set.set_number} ({set.player_side === 'near' ? 'Premier plan' : 'Fond'})
                  </button>
                ))}
              </div>
            </div>
            {currentSetInfo && (
              <div className="set-meta-banner">
                <span>📍 <strong>Position du joueur au Set {currentSetInfo.set_number} :</strong> {currentSetInfo.player_side === 'near' ? 'Joueur au premier plan' : 'Joueur au fond de court'}</span>
                <span>📊 Score du set estimé : <strong>{currentSetInfo.points_won} - {currentSetInfo.points_lost}</strong></span>
              </div>
            )}
          </div>
        )}

        {/* Tableau Cible Principal */}
        <div className="table-card glass-panel">
          <div className="table-header">
            <h3>Détail par Coup Technique (TopSpins, Poussettes, Blocs)</h3>
            <span className="table-subtitle">Rapport d'efficacité et points marqués</span>
          </div>
          {renderStrokesTable(displayStrokes)}
        </div>

        {/* Timeline des rallyes récents */}
        {rallies && rallies.length > 0 && (
          <div className="rallies-section glass-panel">
            <h3 className="section-title">Timeline des Échanges (Rallyes Clés)</h3>
            <p className="section-subtitle">Chaque point séquencé avec la trajectoire et le coup terminal</p>
            <div className="rallies-grid">
              {rallies.slice(0, 12).map((rally) => (
                <div key={rally.rally_number} className={`rally-item ${rally.point_won ? 'won' : 'lost'}`}>
                  <div className="rally-header">
                    <span className="rally-tag">Point #{rally.rally_number} • Set {rally.set_number}</span>
                    <span className={`rally-badge ${rally.point_won ? 'badge-won' : 'badge-lost'}`}>
                      {rally.point_won ? 'Point Remporté' : 'Point Perdu'}
                    </span>
                  </div>
                  <div className="rally-timing">{rally.start_time}s ➔ {rally.end_time}s</div>
                  <div className="rally-sequence">
                    {rally.strokes.map((stk, idx) => (
                      <span key={idx} className={`stroke-chip chip-${stk.outcome}`}>
                        {stk.stroke} {stk.outcome === 'winner' && '🏆'} {stk.outcome === 'unforced_error' && '❌'}
                      </span>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </section>
    )
  }

  return (
    <div className="app-container">
      {/* Navbar / Header */}
      <header className="navbar glass-panel">
        <div className="logo-group">
          <span className="logo-icon">🏓</span>
          <div>
            <h1 className="logo-title">SpinTracker</h1>
            <span className="logo-badge">Match Intelligence AI</span>
          </div>
        </div>
        <nav className="nav-links">
          <a href="#upload" className="nav-link">Importer un Match</a>
          {result && <a href="#results" className="nav-link">Voir Statistiques</a>}
        </nav>
      </header>

      {/* Hero Section */}
      <section className="hero-section">
        <h2 className="hero-title">
          Analysez vos matches comme un <span className="gradient-text">Pro</span>
        </h2>
        <p className="hero-description">
          Déposez votre vidéo de match (10 à 30 min, 2 joueurs). SpinTracker identifie votre position,
          suit automatiquement les <strong>changements de côté à chaque set</strong> et quantifie vos
          <strong> TopSpins</strong>, <strong>Poussettes</strong>, <strong>Blocs</strong>, taux de réussite et <strong>Points Gagnants</strong> !
        </p>
      </section>

      {/* Section Upload & Configuration du joueur */}
      <section className="upload-container glass-panel" id="upload">
        <div className="config-grid">
          {/* Sélection du joueur initial */}
          <div className="config-card">
            <label className="config-label">1. Quel joueur êtes-vous au 1er set ?</label>
            <div className="player-selector">
              <button
                type="button"
                className={`player-btn ${playerSide === 'near' ? 'selected' : ''}`}
                onClick={() => setPlayerSide('near')}
              >
                <span className="player-btn-icon">🧍‍♂️</span>
                <div className="player-btn-content">
                  <strong>Joueur Premier Plan</strong>
                  <span>Proche de la caméra (bas de l'écran)</span>
                </div>
              </button>

              <button
                type="button"
                className={`player-btn ${playerSide === 'far' ? 'selected' : ''}`}
                onClick={() => setPlayerSide('far')}
              >
                <span className="player-btn-icon">🏃‍♂️</span>
                <div className="player-btn-content">
                  <strong>Joueur Fond de Court</strong>
                  <span>Loin de la caméra (haut de l'écran)</span>
                </div>
              </button>
            </div>
            <div className="info-box">
              💡 <em>Les changements de côté à la fin de chaque set sont pris en compte automatiquement pendant toute la durée du match !</em>
            </div>
          </div>

          {/* Déposer la vidéo */}
          <div className="config-card">
            <label className="config-label">2. Fichier Vidéo du match</label>
            <div className="dropzone">
              <input
                type="file"
                id="match-video-input"
                accept="video/*"
                onChange={handleFileChange}
                className="file-input"
              />
              <label htmlFor="match-video-input" className="file-input-label">
                <span className="upload-icon">📹</span>
                {selectedFile ? (
                  <div className="file-selected-info">
                    <span className="file-name">{selectedFile.name}</span>
                    <span className="file-size">({(selectedFile.size / (1024 * 1024)).toFixed(1)} Mo)</span>
                    <span className="change-hint">Cliquer pour changer de vidéo</span>
                  </div>
                ) : (
                  <div>
                    <strong>Cliquez pour choisir votre vidéo de match</strong>
                    <p>Format MP4, MOV, MKV (10 à 30 minutes)</p>
                  </div>
                )}
              </label>
            </div>

            {/* Mode d'analyse */}
            <div className="mode-toggle">
              <label className="mode-option">
                <input
                  type="radio"
                  name="analysis_mode"
                  value="match"
                  checked={analysisMode === 'match'}
                  onChange={() => setAnalysisMode('match')}
                />
                <span>Analyse complète de match (2 joueurs + Sets)</span>
              </label>
              <label className="mode-option">
                <input
                  type="radio"
                  name="analysis_mode"
                  value="clip"
                  checked={analysisMode === 'clip'}
                  onChange={() => setAnalysisMode('clip')}
                />
                <span>Clip isolé (1 seul geste technique)</span>
              </label>
            </div>
          </div>
        </div>

        {/* Bouton d'action */}
        <div className="action-row">
          <button
            type="button"
            onClick={handleUpload}
            disabled={loading || !selectedFile}
            className={`analyze-button ${loading ? 'loading' : ''}`}
          >
            {loading ? (
              <span className="btn-loading-content">
                <span className="spinner"></span>
                Traitement IA en cours (analyse des échanges, détection des coups & points)...
              </span>
            ) : (
              '⚡ Lancer l’Analyse du Match'
            )}
          </button>
        </div>

        {error && (
          <div className="error-banner">
            ⚠️ <strong>Erreur :</strong> {error}
          </div>
        )}
      </section>

      {/* Rendu du Dashboard complet */}
      {renderDashboard()}

      {/* Rendu spécifique en mode clip isolé */}
      {result && result.mode === 'clip' && (
        <section className="clip-result-card glass-panel">
          <h3>Résultat du clip isolé</h3>
          <p>Prédiction : <strong>{result.result.prediction.toUpperCase()}</strong></p>
        </section>
      )}

      {/* Footer */}
      <footer className="footer">
        <p>SpinTracker • Analyse Biomécanique & Tactique de Tennis de Table • Powered by MediaPipe & Machine Learning</p>
      </footer>
    </div>
  )
}

export default App
