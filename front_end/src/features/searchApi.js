import { createApi, fetchBaseQuery } from "@reduxjs/toolkit/query/react";

export const searchApi = createApi({
  reducerPath: "searchApi",
  baseQuery: fetchBaseQuery({ baseUrl: "/api" }),
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
