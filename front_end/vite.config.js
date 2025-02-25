import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { fileURLToPath, URL } from "url";
import tailwindcss from "@tailwindcss/vite"

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@": fileURLToPath(new URL("./src", import.meta.url)),
    },
  },
  server: {
    proxy: {
      // Proxy API requests starting with /api to the Flask backend
      '/api': {
        target: 'http://127.0.0.1:8080', // Flask backend URL
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""), // Remove /api prefix
      },
    },
  },
});
