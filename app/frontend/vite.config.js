import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8081',
        changeOrigin: true,
        timeout:      600_000,  // socket idle (incoming → vite)
        proxyTimeout: 600_000,  // socket idle (vite → backend) — CPU inference can take 2-4 min
      },
    },
  },
})
