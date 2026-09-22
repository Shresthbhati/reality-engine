import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Reality Engine",
  description: "A coherent spatial system for understanding reality.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
