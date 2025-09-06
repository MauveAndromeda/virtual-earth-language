import React from 'react'
import { createRoot } from 'react-dom/client'
import App from './App'
import 'cesium/Build/Cesium/Widgets/widgets.css'
import './styles.css'

const root = createRoot(document.getElementById('root')!)
root.render(<App />)
