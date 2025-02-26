import { useState, useEffect } from "react";
import { useSearchParams, useLocation } from "react-router-dom";
import { useSearchQuery } from "../features/searchApi";
import { Input } from "@/components/ui/input";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Button } from "@/components/ui/button";
import {
  Search,
  X,
  Eye,
  MessageSquare,
  ThumbsUp,
  Filter,
  Star,
} from "lucide-react";

const ResultsPage = () => {
  const [searchParams, setSearchParams] = useSearchParams("python");
  const location = useLocation();

  const initialQuery = decodeURIComponent(searchParams.get("query") || "");
  const [query, setQuery] = useState(initialQuery);
  const [selectedFilters, setSelectedFilters] = useState([]);
  const [currentPage, setCurrentPage] = useState(1);
  const resultsPerPage = 5;

  // Determine if it's an advanced search based on the route
  const isAdvancedSearch = location.pathname.includes("advanced_search");

  // Fetch search results using RTK Query with correct search type
  const { data, error, isLoading, refetch } = useSearchQuery(
    { query, searchType: isAdvancedSearch ? "advanced" : "regular" },
    { skip: !query }
  );

  useEffect(() => {
    if (query.trim()) {
      refetch();
    }
  }, [query, refetch]);

  const handleSearch = () => {
    if (query.trim()) {
      setSearchParams({ query: encodeURIComponent(query) });
      refetch();
    }
  };

  const clearQuery = () => {
    setQuery("");
    setSearchParams({});
  };

  // Pagination logic
  const indexOfLastResult = currentPage * resultsPerPage;
  const indexOfFirstResult = indexOfLastResult - resultsPerPage;
  const currentResults =
    data?.result.slice(indexOfFirstResult, indexOfLastResult) || [];
  const totalPages = Math.ceil((data?.result.length || 0) / resultsPerPage);

  const toggleFilter = (filter) => {
    setSelectedFilters((prevFilters) =>
      prevFilters.includes(filter)
        ? prevFilters.filter((f) => f !== filter)
        : [...prevFilters, filter]
    );
  };

  return (
    <div className="flex flex-col min-h-screen bg-gradient-to-br from-[#F5F7FA] to-[#E0E7EE]">
      <div className="w-full max-w-6xl mx-auto mt-6 flex items-center px-4 space-x-6">
        <h1 className="text-3xl font-semibold text-blue-600 whitespace-nowrap">
          CodeMe
        </h1>
        <div className="relative flex items-center bg-white shadow-md rounded-full w-[75%]">
          <Search className="text-gray-400 ml-4" />
          <Input
            type="text"
            placeholder="Search coding questions..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            className="flex-grow shadow-none border-none focus:outline-none focus:ring-0 focus-visible:ring-0 text-gray-700 px-4"
          />
          {query && (
            <X
              className="text-gray-400 cursor-pointer mx-2"
              onClick={clearQuery}
            />
          )}
          <Button
            className="h-[44px] px-6 bg-blue-500 text-white rounded-full hover:bg-blue-600 transition-all flex items-center justify-center"
            onClick={handleSearch}
          >
            Search
          </Button>
        </div>
      </div>

      <div className="flex flex-grow w-full max-w-6xl mx-auto mt-10 grid grid-cols-3 gap-8 px-4 pb-7">
        <div className="col-span-2 space-y-6">
          {isLoading && <p>Loading results...</p>}
          {error && <p className="text-red-500">Error fetching results.</p>}
          {currentResults.map((result, index) => (
            <div key={index} className="border-b border-gray-300 pb-4">
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger>
                    <a
                      href={`https://stackoverflow.com/questions/${result.doc_id}`}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-lg font-bold text-blue-600 flex items-center space-x-2 hover:underline"
                    >
                      🔵 <span>{result.title}</span>
                    </a>
                  </TooltipTrigger>
                  <TooltipContent>
                    <p className="bg-gray-600 text-white p-2 rounded">
                      Go to the Stack Overflow page
                    </p>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>

              <p className="text-gray-600 mt-1 italic">{result.body}</p>
              <div className="flex space-x-2 mt-2">
                {result.tags.split("|").map(
                  (tag, i) =>
                    tag && (
                      <span
                        key={i}
                        className="bg-yellow-200 text-yellow-700 text-xs font-semibold px-2 py-1 rounded-full flex items-center space-x-1"
                      >
                        ✏️ {tag}
                      </span>
                    )
                )}
              </div>
              <div className="flex items-center space-x-6 mt-2 text-gray-500 text-sm">
                <span
                  className={`flex items-center space-x-1 ${
                    result.score >= 0 ? "text-green-600" : "text-red-600"
                  }`}
                >
                  <ThumbsUp size={16} />
                  <span>
                    {result.score >= 0 ? `+${result.score}` : result.score}{" "}
                    upvotes
                  </span>
                </span>
                <span className="flex items-center space-x-1 text-[#6B7280]">
                  <Eye size={16} />
                  <span>{result.view_count} views</span>
                </span>
                <span className="flex items-center space-x-1 text-[#A855F7]">
                  <MessageSquare size={16} />
                  <span>{result.comment_count} comments</span>
                </span>
                <span className="flex items-center space-x-1 text-[#F97316]">
                  <Star size={16} />
                  <span>{result.favorite_count} favorites</span>
                </span>
              </div>
            </div>
          ))}

          {/* Pagination */}
          {data?.result.length > resultsPerPage && (
            <div className="flex justify-between mt-auto pb-10">
              <Button
                className={`px-4 py-2 rounded-lg ${
                  currentPage === 1
                    ? "opacity-50 cursor-not-allowed"
                    : "bg-blue-500 text-white hover:bg-blue-600"
                }`}
                disabled={currentPage === 1}
                onClick={() => setCurrentPage((prev) => prev - 1)}
              >
                Previous
              </Button>
              <span className="text-gray-700">
                Page {currentPage} of {totalPages}
              </span>
              <Button
                className={`px-4 py-2 rounded-lg ${
                  currentPage === totalPages
                    ? "opacity-50 cursor-not-allowed"
                    : "bg-blue-500 text-white hover:bg-blue-600"
                }`}
                disabled={currentPage === totalPages}
                onClick={() => setCurrentPage((prev) => prev + 1)}
              >
                Next
              </Button>
            </div>
          )}
        </div>

        {/* Filters */}
        <div className="col-span-1">
          <h3 className="text-lg font-bold text-gray-700 flex items-center space-x-2">
            <Filter size={20} className="text-blue-500" />
            <span>Filters</span>
          </h3>
          <div className="mt-4 space-y-3">
            {["Python", "JavaScript", "Web Scraping", "Machine Learning"].map(
              (filter) => (
                <label key={filter} className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    checked={selectedFilters.includes(filter)}
                    onChange={() => toggleFilter(filter)}
                    className="h-4 w-4 text-blue-600"
                  />
                  <span className="text-gray-700">{filter}</span>
                </label>
              )
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default ResultsPage;
