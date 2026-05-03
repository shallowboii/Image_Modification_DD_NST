import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/generate': 'http://localhost:8000',
      '/job':      'http://localhost:8000',
      '/results':  'http://localhost:8000',
    }
  }
})
