import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
export default defineConfig({
  plugins: [react()],
  build: {
    outDir: process.env.AIGC_WEB_ROOT || "/data/zjr/aigc_studio/web-dist",
    emptyOutDir: true,
  },
  server: { proxy: { "/api": "http://127.0.0.1:18790" } },
});
