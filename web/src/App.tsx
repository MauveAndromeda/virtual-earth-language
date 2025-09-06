import React, { useState } from 'react'
import CesiumViewer from './viewer/CesiumViewer'
import LayerPanel from './ui/LayerPanel'
import Inspector from './ui/Inspector'

export type LayerState = { sampleGeoJSON: boolean }

export default function App() {
  const [layers, setLayers] = useState<LayerState>({ sampleGeoJSON: true })
  const [selected, setSelected] = useState<any>(null)

  return (
    <div className="app">
      <header>
        <div>
          <strong>Virtual Earth — Language Emergence</strong>
          <div className="metric">Demo: OSM imagery + sample GeoJSON + pick inspector</div>
        </div>
        <div style={{display:'flex', gap:8}}>
          <a className="btn" href="https://github.com/MauveAndromeda/virtual-earth-language" target="_blank" rel="noreferrer">GitHub</a>
          <a className="btn" href="#" onClick={(e) => { e.preventDefault(); alert('Add more layers via /public/data and src/viewer/CesiumViewer.tsx') }}>How to add data</a>
        </div>
      </header>
      <div className="panel left">
        <LayerPanel layers={layers} onChange={setLayers} />
      </div>
      <div className="panel main">
        <CesiumViewer layers={layers} onSelect={setSelected} />
      </div>
      <div className="panel right">
        <Inspector selected={selected} />
      </div>
    </div>
  )
}
