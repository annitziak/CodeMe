import { useState } from "react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Search, Eye, MessageSquare, ThumbsUp, Filter } from "lucide-react";
import { useSearchParams } from "react-router-dom";

const searchResults = [
  {
    title: "How to use Python for web scraping",
    description:
      "I'm trying to scrape data from a website using BeautifulSoup but facing an issue...",
    tags: ["Python", "Web Scraping"],
    upvotes: 245,
    views: "1.2K",
    comments: 15,
  },
  ...Array.from({ length: 15 }, (_, i) => ({
    title: `Python Web Scraping - Question ${i + 2}`,
    description: `Another question about web scraping techniques and issues...`,
    tags: ["Python", "Web Scraping"],
    upvotes: Math.floor(Math.random() * 500),
    views: `${Math.floor(Math.random() * 3)}K`,
    comments: Math.floor(Math.random() * 20),
  })),
];

const ResultsPage = () => {
  const [selectedFilters, setSelectedFilters] = useState([]);
  const [currentPage, setCurrentPage] = useState(1);
  const resultsPerPage = 5;

  const indexOfLastResult = currentPage * resultsPerPage;
  const indexOfFirstResult = indexOfLastResult - resultsPerPage;
  const currentResults = searchResults.slice(
    indexOfFirstResult,
    indexOfLastResult
  );

  const totalPages = Math.ceil(searchResults.length / resultsPerPage);

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
            className="flex-grow shadow-none border-none focus:outline-none focus:ring-0 focus-visible:ring-0 text-gray-700 px-4"
          />
          <Button className="h-[44px] px-6 bg-blue-500 text-white rounded-full hover:bg-blue-600 transition-all flex items-center justify-center">
            Search
          </Button>
        </div>
      </div>

      <div className="flex flex-grow w-full max-w-6xl mx-auto mt-10 grid grid-cols-3 gap-8 px-4 pb-7">
        <div className="col-span-2 space-y-6">
          {currentResults.map((result, index) => (
            <div key={index} className="border-b border-gray-300 pb-4">
              <h3 className="text-lg font-bold text-blue-600 flex items-center space-x-2">
                🔵 <span>{result.title}</span>
              </h3>
              <p className="text-gray-600 mt-1 italic">{result.description}</p>
              <div className="flex space-x-2 mt-2">
                {result.tags.map((tag, i) => (
                  <span
                    key={i}
                    className="bg-yellow-200 text-yellow-700 text-xs font-semibold px-2 py-1 rounded-full flex items-center space-x-1"
                  >
                    ✏️ {tag}
                  </span>
                ))}
              </div>
              <div className="flex items-center space-x-6 mt-2 text-gray-500 text-sm">
                <span className="flex items-center space-x-1">
                  <ThumbsUp size={16} />
                  <span>{result.upvotes} upvotes</span>
                </span>
                <span className="flex items-center space-x-1">
                  <Eye size={16} />
                  <span>{result.views} views</span>
                </span>
                <span className="flex items-center space-x-1">
                  <MessageSquare size={16} />
                  <span>{result.comments} comments</span>
                </span>
              </div>
            </div>
          ))}

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
        </div>

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
