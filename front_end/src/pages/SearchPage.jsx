import { Input } from "../components/ui/input";
import { Button } from "../components/ui/button";
import { Search } from "lucide-react";

export default function SearchPage() {
  return (
    <div className="flex flex-col h-screen justify-between bg-gradient-to-br from-[#F5F7FA] to-[#E0E7EE]">
      <header className="py-3 px-6 text-gray-600 text-sm">
        About
      </header>
      <main className="flex flex-col items-center justify-center flex-grow text-center px-6 relative">
        <div className="absolute inset-0 flex items-center justify-center opacity-10">
          <Search className="w-[300px] h-[300px] text-gray-400" />
        </div>
        <h1 className="text-4xl font-semibold text-blue-600 mb-2">CodeMe</h1>
        <p className="text-gray-600 mb-6">
          Find code snippets, solutions, and programming insights instantly.
        </p>
        <div className="relative w-full max-w-2xl">
          <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 text-gray-400" />
          <Input
            type="text"
            placeholder="Search for coding questions..."
            className="pl-12 pr-32 py-4 text-lg shadow-md rounded-full border border-gray-300 focus:ring-2 focus:ring-blue-400 focus:outline-none"
          />
          <Button
            className="absolute right-3 top-1/2 transform -translate-y-1/2 bg-blue-500 text-white px-6 py-2 rounded-full hover:bg-blue-600"
          >
            Search
          </Button>
        </div>
      </main>

      <footer className="py-4 bg-gray-200 text-center text-gray-600 text-sm">
        CodeMe © 2025
      </footer>
    </div>
  );
}
