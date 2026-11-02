import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Reality Engine",
  description: "A coherent spatial system for understanding reality.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        {/* Applies the saved theme before first paint so a light/dark choice never flashes. Mirrors lib/prefs.ts. */}
        <script
          dangerouslySetInnerHTML={{
            __html:
              'try{var t=localStorage.getItem("re-theme");if(t==="light"||t==="dark")document.documentElement.setAttribute("data-theme",t)}catch(e){}',
          }}
        />
      </head>
      <body>{children}</body>
    </html>
  );
}
