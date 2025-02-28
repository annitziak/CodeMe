import { createApi, fetchBaseQuery } from "@reduxjs/toolkit/query/react";

const API_BASE_URL = import.meta.env.VITE_API_URL;

export const searchApi = createApi({
  reducerPath: "searchApi",
  // baseQuery: fetchBaseQuery({ baseUrl: "/api" }),
  baseQuery: fetchBaseQuery({ baseUrl: API_BASE_URL }),
  endpoints: (builder) => ({
    search: builder.query({
      query: ({ query, page = 0, page_size = 10 }) => ({
        url: "/search",
        method: "GET",
        params: { query, page, page_size },
      }),
    }),

    searchWithFilters: builder.mutation({
      query: ({ query, page, page_size, filters, searchType }) => ({
        url: searchType === "advanced" ? "/advanced_search" : "/search",
        method: "POST",
        body: { query, page, page_size, filters },
      }),
    }),
  }),
});

export const { useSearchQuery, useSearchWithFiltersMutation } = searchApi;
