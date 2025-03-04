import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Input } from "../components/ui/input";
import { Button } from "../components/ui/button";
import { Search } from "lucide-react";

export default function SearchPage() {
  const [query, setQuery] = useState("");
  const [isAdvancedSearch, setIsAdvancedSearch] = useState(false);
  const navigate = useNavigate();

  const handleSearch = () => {
    if (query.trim()) {
      const searchType = isAdvancedSearch ? "advanced_search" : "results";
      navigate(`/${searchType}?query=${encodeURIComponent(query)}`);
    }
  };

  return (
    <div className="flex flex-col h-screen justify-between bg-gradient-to-br from-[#F5F7FA] to-[#E0E7EE]">
      <header className="py-3 px-6 text-gray-600 text-sm"></header>

      <main className="flex flex-col items-center justify-center flex-grow text-center px-6 relative">
        {/* Background Icon */}
        <div className="absolute inset-0 flex items-center justify-center opacity-10 pointer-events-none">
          <Search className="w-[300px] h-[300px] text-gray-400" />
        </div>

        <h1 className="text-4xl font-semibold text-blue-600 mb-2">CodeMe</h1>
        <p className="text-gray-600 mb-6">
          Find code snippets, solutions, and programming insights instantly.
        </p>

        <div className="flex space-x-4 mb-6">
          <Button
            className={`px-4 py-2 rounded-full w-32 cursor-pointer ${
              !isAdvancedSearch ? "bg-blue-500 text-white" : "bg-gray-200 text-gray-700"
            }`}
            onClick={() => setIsAdvancedSearch(false)}
          >
            Search
          </Button>
          <Button
            className={`px-4 py-2 rounded-full cursor-pointer ${
              isAdvancedSearch ? "bg-blue-500 text-white" : "bg-gray-200 text-gray-700"
            }`}
            onClick={() => setIsAdvancedSearch(true)}
          >
            Advanced Search
          </Button>
        </div>
        <form
          onSubmit={(e) => {
            e.preventDefault();
            handleSearch();
          }}
          className="relative flex items-center bg-white shadow-md rounded-full w-full max-w-2xl px-4 py-2"
        >
          <Search className="text-gray-400 mr-2" />
          <Input
            type="text"
            placeholder={
              isAdvancedSearch
                ? "Advanced search coding questions..."
                : "Search coding questions..."
            }
            className="flex-grow shadow-none border-none focus:outline-none focus:ring-0 focus-visible:ring-0 text-gray-700 px-2"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          <Button
            type="submit"
            className="h-[44px] cursor-pointer px-6 bg-blue-500 text-white rounded-full hover:bg-blue-600 transition-all flex items-center justify-center"
          >
            Search
          </Button>
        </form>
      </main>

      <footer className="py-4 bg-gray-200 text-center text-gray-600 text-sm">
        CodeMe © 2025
      </footer>
    </div>
  );
}
