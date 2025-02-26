import { createApi, fetchBaseQuery } from '@reduxjs/toolkit/query/react';

export const searchApi = createApi({
  reducerPath: 'searchApi',
  baseQuery: fetchBaseQuery({ baseUrl: '/api' }),
  endpoints: (builder) => ({
    search: builder.query({
      query: ({ query, searchType, page = 0, page_size = 20 }) => ({
        url: searchType === "advanced" ? '/advanced_search' : '/search',
        method: 'GET',
        params: { query, page, page_size },
      }),
    }),
  }),
});

export const { useSearchQuery } = searchApi;
