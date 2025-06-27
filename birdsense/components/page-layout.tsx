import type { ReactNode } from "react"

interface PageLayoutProps {
  children: ReactNode
  title?: string
  description?: string
  maxWidth?: "sm" | "md" | "lg" | "xl" | "2xl" | "full"
  padding?: "none" | "sm" | "md" | "lg"
}

export default function PageLayout({
  children,
  title,
  description,
  maxWidth = "2xl",
  padding = "md",
}: PageLayoutProps) {
  const maxWidthClasses = {
    sm: "max-w-sm",
    md: "max-w-md",
    lg: "max-w-lg",
    xl: "max-w-xl",
    "2xl": "max-w-2xl",
    full: "max-w-full",
  }

  const paddingClasses = {
    none: "",
    sm: "p-4",
    md: "p-6 md:p-8",
    lg: "p-8 md:p-12",
  }

  return (
    <div className={`mx-auto ${maxWidthClasses[maxWidth]} ${paddingClasses[padding]}`}>
      {(title || description) && (
        <div className="mb-8 text-center">
          {title && <h1 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">{title}</h1>}
          {description && <p className="text-lg text-gray-600 max-w-2xl mx-auto">{description}</p>}
        </div>
      )}
      {children}
    </div>
  )
}
