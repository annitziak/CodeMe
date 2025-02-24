\set ON_ERROR_STOP on

DROP TABLE IF EXISTS temp_posts CASCADE;
DROP TABLE IF EXISTS temp_answers CASCADE;
DROP TABLE IF EXISTS temp_users CASCADE;
DROP TABLE IF EXISTS temp_posthistory CASCADE;
DROP TABLE IF EXISTS temp_comments;
DROP TABLE IF EXISTS temp_votes;
DROP TABLE IF EXISTS temp_postlinks;
DROP TABLE IF EXISTS temp_badges;

CREATE TABLE temp_posts (LIKE posts INCLUDING ALL);
INSERT INTO temp_posts
SELECT *
FROM posts
WHERE posttypeid = 1
-- ORDER BY creationdate DESC
LIMIT number_of_posts;
-- Could limit to some `creationdate` range

CREATE TABLE temp_answers (LIKE posts INCLUDING ALL);
INSERT INTO temp_answers
SELECT *
FROM posts
WHERE posttypeid = 2 AND parentid IN (SELECT id FROM temp_posts);

INSERT INTO temp_posts
SELECT *
FROM temp_answers;


CREATE TABLE temp_users (LIKE users INCLUDING ALL);
INSERT INTO temp_users
(
SELECT DISTINCT u.*
FROM users u
INNER JOIN temp_posts p ON p.owneruserid = u.id
UNION
SELECT DISTINCT u.*
FROM users u
INNER JOIN temp_posts p ON p.lasteditoruserid = u.id
);

CREATE TABLE temp_posthistory (LIKE posthistory INCLUDING ALL);
INSERT INTO temp_posthistory
SELECT ph.*
FROM posthistory ph
INNER JOIN temp_posts p ON ph.postid = p.id;

CREATE TABLE temp_comments (LIKE comments INCLUDING ALL);
INSERT INTO temp_comments
SELECT c.*
FROM comments c
INNER JOIN temp_posts p ON c.postid = p.id;

CREATE TABLE temp_votes (LIKE votes INCLUDING ALL);
INSERT INTO temp_votes
SELECT v.*
FROM votes v
INNER JOIN temp_posts p ON v.postid = p.id;

CREATE TABLE temp_postlinks (LIKE postlinks INCLUDING ALL);
INSERT INTO temp_postlinks
SELECT DISTINCT pl.*
FROM postlinks pl
INNER JOIN temp_posts p ON pl.postid = p.id
   OR pl.relatedpostid = p.id;

CREATE TABLE temp_badges (LIKE badges INCLUDING ALL);
INSERT INTO temp_badges
SELECT b.*
FROM badges b
INNER JOIN temp_users u ON b.userid = u.id;

"""
ALTER TABLE temp_posts ADD FOREIGN KEY (owneruserid) REFERENCES temp_users(id);
ALTER TABLE temp_posts ADD FOREIGN KEY (lasteditoruserid) REFERENCES temp_users(id);
ALTER TABLE temp_posthistory ADD FOREIGN KEY (postid) REFERENCES temp_posts(id);
ALTER TABLE temp_posthistory ADD FOREIGN KEY (userid) REFERENCES temp_users(id);
ALTER TABLE temp_comments ADD FOREIGN KEY (postid) REFERENCES temp_posts(id);
ALTER TABLE temp_comments ADD FOREIGN KEY (userid) REFERENCES temp_users(id);
ALTER TABLE temp_votes ADD FOREIGN KEY (postid) REFERENCES temp_posts(id);
ALTER TABLE temp_postlinks ADD FOREIGN KEY (postid) REFERENCES temp_posts(id);
ALTER TABLE temp_postlinks ADD FOREIGN KEY (relatedpostid) REFERENCES temp_posts(id);
ALTER TABLE temp_badges ADD FOREIGN KEY (userid) REFERENCES temp_users(id);
"""


\copy (SELECT * FROM temp_users ORDER BY id) TO output_path/users.csv WITH (FORMAT CSV, HEADER);

\copy (SELECT * FROM temp_posts ORDER BY id) TO output_path/posts.csv WITH (FORMAT CSV, HEADER);
\copy (SELECT * FROM temp_posthistory ORDER BY id) TO output_path/posthistory.csv WITH (FORMAT CSV, HEADER);
\copy (SELECT * FROM temp_comments ORDER BY id) TO output_path/comments.csv WITH (FORMAT CSV, HEADER);
\copy (SELECT * FROM temp_votes ORDER BY id) TO output_path/votes.csv WITH (FORMAT CSV, HEADER);
\copy (SELECT * FROM temp_postlinks ORDER BY id) TO output_path/postlinks.csv WITH (FORMAT CSV, HEADER);
\copy (SELECT * FROM temp_badges ORDER BY id) TO output_path/badges.csv WITH (FORMAT CSV, HEADER);
\copy (SELECT * FROM tags ORDER BY id) TO output_path/tags.csv WITH (FORMAT CSV, HEADER);

\! pg_dump --schema-only --table=temp_users --table=temp_posts --table=temp_posthistory --table=temp_comments --table=temp_votes --table=temp_postlinks --table=tags --table=temp_badges --schema stack_overflow --no-owner db_name > output_path/schema.sql

\! sed -i 's/temp_users/users/g; s/temp_posts/posts/g; s/temp_posthistory/posthistory/g; s/temp_comments/comments/g; s/temp_votes/votes/g; s/temp_postlinks/postlinks/g; s/temp_tags/tags/g; s/temp_badges/badges/g;' output_path/schema.sql

DROP TABLE IF EXISTS temp_posts;
DROP TABLE IF EXISTS temp_answers;
DROP TABLE IF EXISTS temp_users;
DROP TABLE IF EXISTS temp_posthistory;
DROP TABLE IF EXISTS temp_comments;
DROP TABLE IF EXISTS temp_votes;
DROP TABLE IF EXISTS temp_postlinks;
DROP TABLE IF EXISTS temp_badges;

COMMIT;
