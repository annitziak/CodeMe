import { useState, useEffect } from "react";
import { useSearchParams, useLocation } from "react-router-dom";
import {
  useSearchQuery,
  useSearchWithFiltersMutation,
} from "../features/searchApi";
import { Input } from "@/components/ui/input";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Search, X, Eye, ThumbsUp, Filter } from "lucide-react";

const ResultsPage = () => {
  const [searchParams, setSearchParams] = useSearchParams();
  const location = useLocation();

  const initialQuery = decodeURIComponent(searchParams.get("query") || "");
  const initialPage = parseInt(searchParams.get("page") || "0", 10);
  const pageSize = 10;

  const [query, setQuery] = useState(initialQuery);
  const [selectedFilters, setSelectedFilters] = useState([]);
  const [usePostResults, setUsePostResults] = useState(false);
  const [sortByDate, setSortByDate] = useState(false);

  const isAdvancedSearch = location.pathname.includes("advanced_search");

  // GET request (only skip if no query)
  const {
    data: getData,
    isLoading: getLoading,
    refetch,
  } = useSearchQuery(
    { query, page: initialPage, page_size: pageSize },
    { skip: !query.trim() }
  );

  // POST request (triggered when filters are applied)
  const [searchWithFilters, { data: postData, isLoading: postLoading }] =
    useSearchWithFiltersMutation();

  useEffect(() => {
    if (!query.trim()) return;

    // If no filters & not sorting by date, do GET refetch
    if (selectedFilters.length === 0 && !sortByDate) {
      setUsePostResults(false);
      refetch();
    } else {
      // Otherwise, do POST
      setUsePostResults(true);
      searchWithFilters({
        query,
        page: initialPage,
        page_size: pageSize,
        searchType: isAdvancedSearch ? "advanced" : "regular",
        filters: {
          tags: selectedFilters,
          date: sortByDate,
        },
      });
    }
  }, [
    query,
    selectedFilters,
    sortByDate,
    initialPage,
    isAdvancedSearch,
    pageSize,
    refetch,
    searchWithFilters,
  ]);

  const handleSearch = () => {
    if (query.trim()) {
      setSearchParams({ query: encodeURIComponent(query), page: 0 });
      if (selectedFilters.length === 0 && !sortByDate) {
        setUsePostResults(false);
        refetch();
      } else {
        setUsePostResults(true);
        searchWithFilters({
          query,
          page: 0,
          page_size: pageSize,
          searchType: isAdvancedSearch ? "advanced" : "regular",
          filters: {
            tags: selectedFilters,
            date: sortByDate,
          },
        });
      }
    }
  };

  const toggleFilter = (filter) => {
    setSelectedFilters((prevFilters) => {
      const newFilters = prevFilters.includes(filter)
        ? prevFilters.filter((f) => f !== filter)
        : [...prevFilters, filter];

      setUsePostResults(newFilters.length > 0);

      // Wait for the state to update before making API calls
      setTimeout(() => {
        if (newFilters.length === 0) {
          refetch();
        } else {
          searchWithFilters({
            query,
            page: initialPage,
            page_size: pageSize,
            searchType: isAdvancedSearch ? "advanced" : "regular",
            filters: {
              tags: newFilters,
            },
          });
        }
      }, 0);

      return newFilters;
    });
  };

  const results = usePostResults ? postData || [] : getData || [];
  const isLoading = usePostResults ? postLoading : getLoading;

  const clearQuery = () => {
    setQuery("");
    setSearchParams({});
  };

  const handleNextPage = () => {
    if (results?.has_next) {
      setSearchParams({
        query: encodeURIComponent(query),
        page: results.page + 1,
      });
    }
  };

  const handlePrevPage = () => {
    if (results?.has_prev) {
      setSearchParams({
        query: encodeURIComponent(query),
        page: results.page - 1,
      });
    }
  };


  return (
    <div className="flex flex-col min-h-screen bg-gradient-to-br from-[#F5F7FA] to-[#E0E7EE]">
      <div className="w-full mx-auto mt-6 px-12 space-y-3 lg:space-y-0 lg:flex lg:items-center lg:space-x-6">
        <h1 className="text-3xl font-semibold text-blue-600 text-center lg:text-left">
          CodeMe
        </h1>
        <div className="relative flex items-center bg-white shadow-md rounded-full w-full lg:w-[75%]">
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

      {/* Mobile Filters */}
      <div className="w-full mt-4 lg:hidden px-3">
        <h3 className="text-lg font-bold text-gray-700 flex items-center space-x-2">
          <Filter size={20} className="text-blue-500" />
          <span>Filters</span>
        </h3>
        <div className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-3">
          {[
            "Programming & Development Fundamentals",
            "Software Engineering & System Design",
            "Advanced Computing & Algorithms",
            "Technologies & Frameworks",
            "Other",
          ].map((filter) => (
            <label key={filter} className="flex items-center space-x-2">
              <Checkbox
                checked={selectedFilters.includes(filter)}
                onCheckedChange={() => toggleFilter(filter)}
              />
              <span className="text-gray-700">{filter}</span>
            </label>
          ))}
        </div>
        <div className="flex items-center space-x-2 mt-3">
          <Checkbox
            id="terms"
            checked={sortByDate}
            onCheckedChange={() => setSortByDate((prev) => !prev)}
          />
          <label htmlFor="terms" className="text-gray-700">
            Sort by date
          </label>
        </div>
      </div>

      {isLoading ? (
        <p className="text-blue-500 text-center mt-5 text-3xl">
          Loading results...
        </p>
      ) : (
        <div className="flex flex-grow w-full max-w-7xl mx-auto mt-10 gap-8 pb-7 px-4">
          <div className="w-[90%]">
            {results?.result?.map((result, index) => (
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
                          className="text-center bg-yellow-200 text-yellow-700 text-xs font-semibold px-2 py-1 rounded-full flex items-center space-x-1"
                        >
                          ✏️ {tag}
                        </span>
                      )
                  )}
                </div>
                <div className="flex items-center space-x-6 mt-2 text-gray-500 text-sm">
                  <span className="flex items-center space-x-1 text-green-600">
                    <ThumbsUp size={16} />
                    <span>{result.score} upvotes</span>
                  </span>
                  <span className="flex items-center space-x-1 text-[#6B7280]">
                    <Eye size={16} />
                    <span>{result.view_count} views</span>
                  </span>
                </div>
              </div>
            ))}

            <div className="flex justify-between items-center mt-3 pb-10">
              <Button
                className={`px-4 py-2 rounded-lg ${
                  !results?.has_prev
                    ? "opacity-50 cursor-not-allowed"
                    : "bg-blue-500 text-white hover:bg-blue-600"
                }`}
                disabled={!results?.has_prev}
                onClick={handlePrevPage}
              >
                Previous
              </Button>
              <span className="text-gray-700">Page {results?.page + 1}</span>
              <Button
                className={`px-4 py-2 rounded-lg ${
                  !results?.has_next
                    ? "opacity-50 cursor-not-allowed"
                    : "bg-blue-500 text-white hover:bg-blue-600"
                }`}
                disabled={!results?.has_next}
                onClick={handleNextPage}
              >
                Next
              </Button>
            </div>
          </div>

          {/* Desktop Filters */}
          <div className="hidden lg:block">
            <h3 className="text-lg font-bold text-gray-700 flex items-center space-x-2">
              <Filter size={20} className="text-blue-500" />
              <span>Filters</span>
            </h3>
            <div className="mt-4 space-y-3 col-span-2">
              {[
                "Programming & Development Fundamentals",
                "Software Engineering & System Design",
                "Advanced Computing & Algorithms",
                "Technologies & Frameworks",
                "Other",
              ].map((filter) => (
                <label key={filter} className="flex items-center space-x-2">
                  <Checkbox
                    checked={selectedFilters.includes(filter)}
                    onCheckedChange={() => toggleFilter(filter)}
                  />
                  <span className="text-gray-700">{filter}</span>
                </label>
              ))}
            </div>
            <div className="flex items-center space-x-2 mt-3">
              <Checkbox
                id="terms"
                checked={sortByDate}
                onCheckedChange={() => setSortByDate((prev) => !prev)}
              />
              <label htmlFor="terms" className="text-gray-700">
                Sort by date
              </label>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ResultsPage;
