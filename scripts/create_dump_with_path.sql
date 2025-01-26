\set ON_ERROR_STOP on

DROP TABLE IF EXISTS public.users CASCADE;
DROP TABLE IF EXISTS public.posts CASCADE;
DROP TABLE IF EXISTS public.comments CASCADE;
DROP TABLE IF EXISTS public.votes CASCADE;
DROP TABLE IF EXISTS public.postlinks CASCADE;
DROP TABLE IF EXISTS public.posthistory CASCADE;
DROP TABLE IF EXISTS public.badges CASCADE;
DROP TABLE IF EXISTS public.tags CASCADE;


BEGIN;
\i stack_overflow_dump_10000/schema.sql

\copy public.posts FROM stack_overflow_dump_10000/posts.csv WITH (FORMAT CSV, HEADER);
\copy public.users FROM stack_overflow_dump_10000/users.csv WITH (FORMAT CSV, HEADER);
\copy public.comments FROM stack_overflow_dump_10000/comments.csv WITH (FORMAT CSV, HEADER);
\copy public.votes FROM stack_overflow_dump_10000/votes.csv WITH (FORMAT CSV, HEADER);
\copy public.tags FROM stack_overflow_dump_10000/tags.csv WITH (FORMAT CSV, HEADER);
\copy public.posthistory FROM stack_overflow_dump_10000/posthistory.csv WITH (FORMAT CSV, HEADER);
\copy public.postlinks FROM stack_overflow_dump_10000/postlinks.csv WITH (FORMAT CSV, HEADER);
\copy public.badges FROM stack_overflow_dump_10000/badges.csv WITH (FORMAT CSV, HEADER);
END;

ALTER TABLE public.badges SET SCHEMA public;
ALTER TABLE public.comments SET SCHEMA public;
ALTER TABLE public.posthistory SET SCHEMA public;
ALTER TABLE public.postlinks SET SCHEMA public;
ALTER TABLE public.posts SET SCHEMA public;
ALTER TABLE public.tags SET SCHEMA public;
