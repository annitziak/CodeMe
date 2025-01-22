\set ON_ERROR_STOP on

BEGIN;
\i :input_path/schema.sql

\copy users FROM :input_path/users.csv WITH (FORMAT CSV, HEADER);
\copy posts FROM :input_path/posts.csv WITH (FORMAT CSV, HEADER);
\copy comments FROM :input_path/comments.csv WITH (FORMAT CSV, HEADER);
\copy votes FROM :input_path/votes.csv WITH (FORMAT CSV, HEADER);
\copy tags FROM :input_path/tags.csv WITH (FORMAT CSV, HEADER);
\copy posthistory FROM :input_path/posthistory.csv WITH (FORMAT CSV, HEADER);
\copy postlinks FROM :input_path/postlinks.csv WITH (FORMAT CSV, HEADER);
\copy badges FROM :input_path/badges.csv WITH (FORMAT CSV, HEADER);
