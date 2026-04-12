import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "IRECS | Research Tool",
  description: "Next-gen evolutionary algorithm interface",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
