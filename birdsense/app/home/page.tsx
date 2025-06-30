"use client"

import { useState, useEffect, useCallback, useRef } from "react"
import { Card, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Search, Clock, Camera, TrendingUp, Loader2 } from "lucide-react"
import AppLayout from "@/components/app-layout"
import PageLayout from "@/components/page-layout"

export default function HomePage() {
  type bird = {
    id: string;
    species: string;
    scientificName: string;
    confidence: number;
    timestamp: string;
    image: string;
    location: string;
    rarity: string;
  };

  type apiResponse = {
    nextCursor: string | null;
    items: bird[];
  };

  const [birds, setBirds] = useState<any[]>([])
  const [searchTerm, setSearchTerm] = useState("")
  const [sortBy, setSortBy] = useState("timestamp")
  const [filterBy, setFilterBy] = useState("all")
  const [loading, setLoading] = useState(false)
  const [hasMore, setHasMore] = useState(true)
  const [cursor, setCursor] = useState<string | null>(null);
  const [initialLoading, setInitialLoading] = useState(true)
  const pageSize = 12; 

  const observer = useRef<IntersectionObserver>()
  const lastBirdElementRef = useCallback(
    (node: HTMLDivElement | null) => {
      if (loading || initialLoading) return
      if (observer.current) observer.current.disconnect()
      observer.current = new IntersectionObserver((entries) => {
        if (entries[0].isIntersecting && hasMore) {
          loadMoreBirds()
        }
      })
      if (node) observer.current.observe(node)
    },
    [loading, hasMore],
  )

  // Load initial data
  useEffect(() => {
    loadInitialBirds()
  }, [])

  async function fetchBirdClassifications(cursor: string | null = null, limit = 12): Promise<apiResponse> {
    const url = new URL("https://yh5oyjgccj.execute-api.us-east-2.amazonaws.com/default/load10");
    url.searchParams.set("limit", String(limit));
    if (cursor) {
      url.searchParams.set("cursor", cursor);
    }

    const res = await fetch(url.toString());

    return res.json();
  }

  const loadInitialBirds = async () => {
    setInitialLoading(true)
    try {
      const {items, nextCursor} = await fetchBirdClassifications(null,pageSize)
      setBirds(items);
      setCursor(nextCursor);
      setHasMore(items.length === pageSize);
    } catch (error) {
      console.error("Failed to load initial birds:", error)
    } finally {
      setInitialLoading(false)
    }
  }

  const loadMoreBirds = async () => {
    if (loading || !hasMore) return

    setLoading(true)
    try {
      const {items, nextCursor} = await fetchBirdClassifications(cursor, pageSize)
      setBirds((prev) => [...prev, ...items]);
      setCursor(nextCursor);
      setHasMore(items.length === pageSize);
    } catch (error) {
      console.error("Failed to load more birds:", error)
    } finally {
      setLoading(false)
    }
  }

  const formatTime = (timestamp: string) => {
    const date = new Date(timestamp)
    const now = new Date()
    const diffInMinutes = Math.floor((now.getTime() - date.getTime()) / (1000 * 60))

    if (diffInMinutes < 60) {
      return `${diffInMinutes}m ago`
    } else if (diffInMinutes < 1440) {
      return `${Math.floor(diffInMinutes / 60)}h ago`
    } else {
      return `${Math.floor(diffInMinutes / 1440)}d ago`
    }
  }

  const getRarityColor = (rarity: string) => {
    switch (rarity) {
      case "Very Common":
        return "bg-green-100 text-green-800 border-green-200"
      case "Common":
        return "bg-blue-100 text-blue-800 border-blue-200"
      case "Uncommon":
        return "bg-yellow-100 text-yellow-800 border-yellow-200"
      case "Rare":
        return "bg-red-100 text-red-800 border-red-200"
      default:
        return "bg-gray-100 text-gray-800 border-gray-200"
    }
  }

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 95) return "text-green-600 bg-green-50"
    if (confidence >= 90) return "text-blue-600 bg-blue-50"
    if (confidence >= 85) return "text-yellow-600 bg-yellow-50"
    return "text-red-600 bg-red-50"
  }

  const filteredAndSortedBirds = birds
    .filter((bird) => {
      const matchesSearch =
        bird.species.toLowerCase().includes(searchTerm.toLowerCase()) ||
        bird.scientificName.toLowerCase().includes(searchTerm.toLowerCase())
      const matchesFilter = filterBy === "all" || bird.rarity.toLowerCase().replace(" ", "") === filterBy
      return matchesSearch && matchesFilter
    })
    .sort((a, b) => {
      switch (sortBy) {
        case "confidence":
          return b.confidence - a.confidence
        case "species":
          return a.species.localeCompare(b.species)
        case "timestamp":
        default:
          return new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
      }
    })

  if (initialLoading) {
    return (
      <AppLayout showNavigation={true}>
        <PageLayout maxWidth="full">
          <div className="flex items-center justify-center min-h-[60vh]">
            <div className="text-center space-y-4">
              <Loader2 className="w-12 h-12 animate-spin text-emerald-500 mx-auto" />
              <div>
                <h3 className="text-lg font-medium text-gray-900">Loading Bird Classifications</h3>
                <p className="text-gray-500">Fetching the latest detections from birdSense AI...</p>
              </div>
            </div>
          </div>
        </PageLayout>
      </AppLayout>
    )
  }

  return (
    <AppLayout showNavigation={true}>
      <PageLayout maxWidth="full">
        {/* Stats Overview */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
          <Card className="bg-gradient-to-r from-emerald-500 to-emerald-600 text-white">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-emerald-100 text-sm font-medium">Total Detections</p>
                  <p className="text-3xl font-bold">{birds.length}+</p>
                </div>
                <Camera className="w-8 h-8 text-emerald-200" />
              </div>
            </CardContent>
          </Card>

          <Card className="bg-gradient-to-r from-blue-500 to-blue-600 text-white">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-blue-100 text-sm font-medium">Species Identified</p>
                  <p className="text-3xl font-bold">{new Set(birds.map((b) => b.species)).size}</p>
                </div>
                <TrendingUp className="w-8 h-8 text-blue-200" />
              </div>
            </CardContent>
          </Card>

          <Card className="bg-gradient-to-r from-purple-500 to-purple-600 text-white">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-purple-100 text-sm font-medium">Average Confidence</p>
                  <p className="text-3xl font-bold">
                    {birds.length > 0
                      ? Math.round(birds.reduce((acc, bird) => acc + bird.confidence, 0) / birds.length)
                      : 0}
                    %
                  </p>
                </div>
                <Clock className="w-8 h-8 text-purple-200" />
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Search and Filter Controls */}
        <Card className="mb-8">
          <CardContent className="p-6">
            <div className="flex flex-col md:flex-row gap-4">
              <div className="relative flex-1">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
                <Input
                  placeholder="Search by species name..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="pl-10"
                />
              </div>

              <Select value={sortBy} onValueChange={setSortBy}>
                <SelectTrigger className="w-full md:w-48">
                  <SelectValue placeholder="Sort by" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="timestamp">Most Recent</SelectItem>
                  <SelectItem value="confidence">Highest Confidence</SelectItem>
                  <SelectItem value="species">Species Name</SelectItem>
                </SelectContent>
              </Select>

              <Select value={filterBy} onValueChange={setFilterBy}>
                <SelectTrigger className="w-full md:w-48">
                  <SelectValue placeholder="Filter by rarity" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Species</SelectItem>
                  <SelectItem value="verycommon">Very Common</SelectItem>
                  <SelectItem value="common">Common</SelectItem>
                  <SelectItem value="uncommon">Uncommon</SelectItem>
                  <SelectItem value="rare">Rare</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </CardContent>
        </Card>

        {/* Bird Classifications Grid with Infinite Scroll */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
          {filteredAndSortedBirds.map((bird, index) => {
            const isLast = index === filteredAndSortedBirds.length - 1
            return (
              <Card
                key={bird.id}
                ref={isLast ? lastBirdElementRef : null}
                className="group hover:shadow-xl transition-all duration-300 hover:-translate-y-2 bg-white border-0 shadow-md"
              >
                <CardContent className="p-0">
                  {/* Bird Image */}
                  <div className="relative overflow-hidden rounded-t-lg">
                    <img
                      src={bird.image || "/placeholder.svg"}
                      alt={bird.species}
                      className="w-full h-56 object-cover group-hover:scale-110 transition-transform duration-500"
                      loading="lazy"
                    />
                    <div className="absolute top-3 right-3">
                      <Badge className={`${getRarityColor(bird.rarity)} border font-medium`}>{bird.rarity}</Badge>
                    </div>
                    <div className="absolute bottom-3 left-3">
                      <Badge variant="secondary" className="bg-black/80 text-white border-0">
                        <Clock className="w-3 h-3 mr-1" />
                        {formatTime(bird.timestamp)}
                      </Badge>
                    </div>
                    <div className="absolute inset-0 bg-gradient-to-t from-black/20 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
                  </div>

                  {/* Bird Information */}
                  <div className="p-5 space-y-4">
                    <div className="space-y-2">
                      <h3 className="font-bold text-xl text-gray-900 group-hover:text-emerald-600 transition-colors leading-tight">
                        {bird.species}
                      </h3>
                      <p className="text-sm text-gray-600 italic font-medium">{bird.scientificName}</p>
                    </div>

                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-2">
                        <span className="text-sm text-gray-500 font-medium">Confidence:</span>
                        <Badge className={`${getConfidenceColor(bird.confidence)} border-0 font-bold`}>
                          {bird.confidence}%
                        </Badge>
                      </div>
                    </div>

                    <div className="flex items-center text-sm text-gray-600 bg-gray-50 rounded-lg px-3 py-2">
                      <span className="truncate font-medium">{bird.location}</span>
                    </div>

                    <Button
                      variant="outline"
                      size="sm"
                      className="w-full group-hover:bg-emerald-500 group-hover:border-emerald-500 group-hover:text-white transition-all duration-300 font-medium"
                    >
                      View Details
                    </Button>
                  </div>
                </CardContent>
              </Card>
            )
          })}
        </div>

        {/* Loading Indicator */}
        {loading && (
          <div className="flex items-center justify-center py-12">
            <div className="text-center space-y-4">
              <Loader2 className="w-8 h-8 animate-spin text-emerald-500 mx-auto" />
              <p className="text-gray-600 font-medium">Loading more bird classifications...</p>
            </div>
          </div>
        )}

        {/* End of Results */}
        {!hasMore && birds.length > 0 && (
          <div className="text-center py-12">
            <div className="space-y-2">
              <p className="text-gray-600 font-medium">You've reached the end of the classifications!</p>
              <p className="text-sm text-gray-500">Showing all {filteredAndSortedBirds.length} bird detections</p>
            </div>
          </div>
        )}

        {/* Empty State */}
        {filteredAndSortedBirds.length === 0 && !loading && (
          <Card className="text-center py-16">
            <CardContent>
              <div className="text-gray-400 mb-6">
                <Search className="w-16 h-16 mx-auto" />
              </div>
              <h3 className="text-xl font-bold text-gray-900 mb-3">No birds found</h3>
              <p className="text-gray-500 mb-6">Try adjusting your search or filter criteria.</p>
              <Button
                onClick={() => {
                  setSearchTerm("")
                  setFilterBy("all")
                }}
                variant="outline"
              >
                Clear Filters
              </Button>
            </CardContent>
          </Card>
        )}
      </PageLayout>
    </AppLayout>
  )
}
