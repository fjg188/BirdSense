import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Bird, Camera, Cpu, Eye } from "lucide-react"
import Link from "next/link"

export default function landingPage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-green-50 via-blue-50 to-emerald-50 flex items-center justify-center p-4">
      <div className="w-full max-w-md space-y-8">
        {/* Logo and App Name */}
        <div className="text-center space-y-4">
          <div className="flex justify-center">
            <div className="relative">
              <div className="w-20 h-20 bg-gradient-to-br from-emerald-500 to-blue-600 rounded-full flex items-center justify-center shadow-lg">
                <Bird className="w-10 h-10 text-white" />
              </div>
              <div className="absolute -bottom-1 -right-1 w-6 h-6 bg-orange-500 rounded-full flex items-center justify-center">
                <Eye className="w-3 h-3 text-white" />
              </div>
            </div>
          </div>
          <div>
            <h1 className="text-4xl font-bold text-blue-700 tracking-tight">
              Bird<span className="text-emerald-600">Sense</span>
            </h1>
            <p className="text-gray-600 mt-2">AI-powered bird species classification</p>
          </div>
        </div>

        {/* Features Preview */}
        <div className="flex justify-center space-x-6 text-center">
          <div className="flex flex-col items-center space-y-2">
            <div className="w-12 h-12 bg-blue-100 rounded-full flex items-center justify-center">
              <Camera className="w-6 h-6 text-blue-600" />
            </div>
            <span className="text-sm text-gray-600">Live Camera</span>
          </div>
          <div className="flex flex-col items-center space-y-2">
            <div className="w-12 h-12 bg-purple-100 rounded-full flex items-center justify-center">
              <Cpu className="w-6 h-6 text-purple-600" />
            </div>
            <span className="text-sm text-gray-600">ML Classification</span>
          </div>
          <div className="flex flex-col items-center space-y-2">
            <div className="w-12 h-12 bg-green-100 rounded-full flex items-center justify-center">
              <Bird className="w-6 h-6 text-green-600" />
            </div>
            <span className="text-sm text-gray-600">Species ID</span>
          </div>
        </div>

        {/* Access Card */}
        <Card className="shadow-xl border-0 bg-white/80 backdrop-blur-sm">
          <CardHeader className="text-center pb-4">
            <CardTitle className="text-xl text-gray-900">Access birdSense</CardTitle>
            <CardDescription>Enter password to view live bird classifications</CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* Password Input */}
            <div className="space-y-2">
              <Input
                type="password"
                placeholder="Enter password"
                className="h-12 text-center text-lg border-2 
                border-gray-300 
                hover:border-emerald-500
                focus-visible:border-emerald-500 
                focus-visible:ring-transparent"
              />
            </div>

            {/* Action Buttons */}
            <div className="grid grid-cols-2 gap-3">
              <Link href="/access" passHref>
              <Button
                variant="outline"
                className="cursor-pointer h-12 border-2 
                border-gray-300
                hover:border-emerald-500 
                hover:text-emerald-600 transition-colors"
              >
                Contact for Access
              </Button>
              </Link>
              <Link href="/about" passHref>
              <Button className="cursor-pointer h-12 bg-gradient-to-r 
              from-emerald-500 to-blue-600 
              hover:from-emerald-600 
              hover:to-blue-700 
              text-white shadow-lg transition-all duration-200 hover:shadow-xl">
                See How It Works
              </Button>
              </Link>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
