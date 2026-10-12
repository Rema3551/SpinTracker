import { useState } from 'react'
import axios from 'axios'
import './App.css'

function App() {
  const [selectedFile, setSelectedFile] = useState(null)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const handleFileChange = (e) => {
    setSelectedFile(e.target.files[0])
    setResult(null)
    setError('')
  }

  const handleUpload = async () => {
    if (!selectedFile) return
    setLoading(true)
    setError('')
    const formData = new FormData()
    formData.append('file', selectedFile)
    const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'
    try {
      const response = await axios.post(`${API_BASE_URL}/upload`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      })
      setResult(response.data)
    } catch (err) {
      setError(err.response?.data?.detail || 'Upload failed')
    } finally {
      setLoading(false)
    }
  }

  const renderMetrics = () => {
    if (!result) return null
    const { metrics, prediction, probabilities } = result
    const probEntries = Object.entries(probabilities)
    return (
      <div className="dashboard">
        <h2 className="title">Résultats de l'analyse</h2>
        <div className="card prediction-card">
          <h3>Prédit : {prediction.toUpperCase()}</h3>
        </div>
        <div className="grid">
          {probEntries.map(([cls, prob]) => (
            <div key={cls} className="card prob-card">
              <span className="label">{cls}</span>
              <div className="bar-container">
                <div className="bar" style={{ width: `${prob * 100}%` }} />
              </div>
              <span className="percentage">{(prob * 100).toFixed(1)}%</span>
            </div>
          ))}
        </div>
        <div className="grid metrics-grid">
          {Object.entries(metrics).map(([k, v]) => (
            <div key={k} className="card metric-card">
              <span className="label">{k.replace('_', ' ')}</span>
              <span className="value">{Number(v).toFixed(3)}</span>
            </div>
          ))}
        </div>
      </div>
    )
  }

  return (
    <div className="app-container">
      <header className="header">
        <h1>SpinTracker Dashboard</h1>
        <p>Upload un match (10‑30 min) et obtenez les statistiques de vos coups.</p>
      </header>
      <section className="upload-section">
        <input type="file" accept="video/*" onChange={handleFileChange} />
        <button onClick={handleUpload} disabled={loading || !selectedFile} className="upload-btn">
          {loading ? 'Analyse...' : 'Analyser la vidéo'}
        </button>
        {error && <div className="error">{error}</div>}
      </section>
      {result && renderMetrics()}
    </div>
  )
}

export default App

