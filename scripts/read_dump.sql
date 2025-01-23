\set ON_ERROR_STOP on

DROP TABLE IF EXISTS public.users;
DROP TABLE IF EXISTS public.posts;
DROP TABLE IF EXISTS public.comments;
DROP TABLE IF EXISTS public.votes;
DROP TABLE IF EXISTS public.postlinks;
DROP TABLE IF EXISTS public.posthistory;
DROP TABLE IF EXISTS public.badges;
DROP TABLE IF EXISTS public.tags;


BEGIN;
\i input_path/schema.sql

\copy public.users FROM input_path/users.csv WITH (FORMAT CSV, HEADER);
\copy public.posts FROM input_path/posts.csv WITH (FORMAT CSV, HEADER);
\copy public.comments FROM input_path/comments.csv WITH (FORMAT CSV, HEADER);
\copy public.votes FROM input_path/votes.csv WITH (FORMAT CSV, HEADER);
\copy public.tags FROM input_path/tags.csv WITH (FORMAT CSV, HEADER);
\copy public.posthistory FROM input_path/posthistory.csv WITH (FORMAT CSV, HEADER);
\copy public.postlinks FROM input_path/postlinks.csv WITH (FORMAT CSV, HEADER);
\copy public.badges FROM input_path/badges.csv WITH (FORMAT CSV, HEADER);
END;

ALTER TABLE public.badges SET SCHEMA public;
ALTER TABLE public.comments SET SCHEMA public;
ALTER TABLE public.posthistory SET SCHEMA public;
ALTER TABLE public.postlinks SET SCHEMA public;
ALTER TABLE public.posts SET SCHEMA public;
ALTER TABLE public.tags SET SCHEMA public;
