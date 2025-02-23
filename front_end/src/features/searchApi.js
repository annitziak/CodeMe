import { createApi, fetchBaseQuery } from '@reduxjs/toolkit/query/react';

export const searchApi = createApi({
  reducerPath: 'searchApi',
  baseQuery: fetchBaseQuery({ baseUrl: 'http://localhost:8080' }),
  endpoints: (builder) => ({
    search: builder.query({
      query: (query) => ({
        url: '/search',
        method: 'GET',
        params: { query },
      }),
    }),
  }),
});

export const { useSearchQuery } = searchApi;
