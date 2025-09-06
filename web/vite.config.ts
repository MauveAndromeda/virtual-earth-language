import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: { port: 5173 },
  build: { sourcemap: false },
  define: {
    CESIUM_BASE_URL: JSON.stringify('/cesium')
  }
})
