import type { Metadata } from "next";
import { Inter, Space_Grotesk, JetBrains_Mono } from "next/font/google";
import "./globals.css";

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
  display: "swap",
});

const spaceGrotesk = Space_Grotesk({
  variable: "--font-space-grotesk",
  subsets: ["latin"],
  display: "swap",
});

const jetbrainsMono = JetBrains_Mono({
  variable: "--font-jetbrains-mono",
  subsets: ["latin"],
  display: "swap",
});

export const metadata: Metadata = {
  title: "NOVA-VAD — Explainable Voice Activity Detection for Noisy Audio",
  description:
    "NOVA-VAD is a lightweight, noise-robust, explainable voice activity detector. 93% accuracy on noisy real-world audio, no GPU required, fully open source.",
  metadataBase: new URL("https://nova-vad.vercel.app"),
  openGraph: {
    title: "NOVA-VAD — Explainable Voice Activity Detection",
    description:
      "Noise-robust, optimized, explainable VAD. 93% accuracy on noisy audio, beating WebRTC, Pyannote, and Silero VAD — with no GPU required.",
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    site: "@Nova_vad",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${inter.variable} ${spaceGrotesk.variable} ${jetbrainsMono.variable} h-full antialiased`}
    >
      <body className="min-h-full flex flex-col bg-background text-foreground">
        {children}
      </body>
    </html>
  );
}
