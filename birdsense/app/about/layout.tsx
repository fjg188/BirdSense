import type React from "react"
import { Analytics } from "@vercel/analytics/next"


export default function HowItWorksLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (<>{children}
    <Analytics /></>
  )
}
