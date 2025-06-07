
import { defineConfig } from 'vite'

export default defineConfig({
  // Minimal Vite config for compatibility
  // The actual app runs on Python Gradio
  build: {
    outDir: 'dist',
  },
  server: {
    port: 3000,
  },
})
