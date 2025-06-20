import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'; // ✅ correct

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    react()],
})
