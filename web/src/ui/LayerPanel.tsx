import React from 'react'
import { LayerState } from '../App'

type Props = { layers: LayerState; onChange: (s: LayerState) => void }

export default function LayerPanel({ layers, onChange }: Props) {
  return (
    <div>
      <h3 style={{marginTop:0}}>Layers</h3>
      <label style={{display:'flex', gap:8, alignItems:'center'}}>
        <input type="checkbox" checked={layers.sampleGeoJSON}
          onChange={(e) => onChange({ ...layers, sampleGeoJSON: e.target.checked })} />
        <span>Sample GeoJSON</span>
      </label>
      <hr style={{opacity:0.2, margin:'12px 0'}}/>
      <p style={{fontSize:12, opacity:0.8}}>
        Add more layers in <code>public/data</code> and wire them in <code>src/viewer/CesiumViewer.tsx</code>.
      </p>
    </div>
  )
}
