import { createApi, fetchBaseQuery } from '@reduxjs/toolkit/query/react';

export const searchApi = createApi({
  reducerPath: 'searchApi',
  baseQuery: fetchBaseQuery({ baseUrl: '/api' }),
  endpoints: (builder) => ({
    search: builder.query({
      // Accept both query and searchType
      query: ({ query, searchType }) => ({
        url: searchType === "advanced" ? '/advanced_search' : '/search',
        method: 'GET',
        params: { query },
      }),
    }),
  }),
});

export const { useSearchQuery } = searchApi;
