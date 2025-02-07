\set ON_ERROR_STOP on

DROP TABLE IF EXISTS temp_posts;
DROP TABLE IF EXISTS temp_users;
DROP TABLE IF EXISTS temp_posthistory;
DROP TABLE IF EXISTS temp_comments;
DROP TABLE IF EXISTS temp_votes;
DROP TABLE IF EXISTS temp_postlinks;
DROP TABLE IF EXISTS temp_badges;

CREATE TABLE temp_posts (LIKE posts INCLUDING ALL);
INSERT INTO temp_posts
SELECT *
FROM posts
-- ORDER BY creationdate DESC
LIMIT 1000000;
-- Could limit to some `creationdate` range

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


\copy (SELECT * FROM temp_users ORDER BY id) TO /media/seanleishman/Disk/stack_overflow_dump_1M/users.csv WITH (FORMAT CSV, HEADER);

\copy (SELECT * FROM temp_posts ORDER BY id) TO /media/seanleishman/Disk/stack_overflow_dump_1M/posts.csv WITH (FORMAT CSV, HEADER);
\copy (SELECT * FROM temp_posthistory ORDER BY id) TO /media/seanleishman/Disk/stack_overflow_dump_1M/posthistory.csv WITH (FORMAT CSV, HEADER);
\copy (SELECT * FROM temp_comments ORDER BY id) TO /media/seanleishman/Disk/stack_overflow_dump_1M/comments.csv WITH (FORMAT CSV, HEADER);
\copy (SELECT * FROM temp_votes ORDER BY id) TO /media/seanleishman/Disk/stack_overflow_dump_1M/votes.csv WITH (FORMAT CSV, HEADER);
\copy (SELECT * FROM temp_postlinks ORDER BY id) TO /media/seanleishman/Disk/stack_overflow_dump_1M/postlinks.csv WITH (FORMAT CSV, HEADER);
\copy (SELECT * FROM temp_badges ORDER BY id) TO /media/seanleishman/Disk/stack_overflow_dump_1M/badges.csv WITH (FORMAT CSV, HEADER);
\copy (SELECT * FROM tags ORDER BY id) TO /media/seanleishman/Disk/stack_overflow_dump_1M/tags.csv WITH (FORMAT CSV, HEADER);

\! pg_dump --schema-only --table=temp_users --table=temp_posts --table=temp_posthistory --table=temp_comments --table=temp_votes --table=temp_postlinks --table=tags --table=temp_badges --schema stack_overflow --no-owner stack_overflow > /media/seanleishman/Disk/stack_overflow_dump_1M/schema.sql

\! sed -i 's/temp_users/users/g; s/temp_posts/posts/g; s/temp_posthistory/posthistory/g; s/temp_comments/comments/g; s/temp_votes/votes/g; s/temp_postlinks/postlinks/g; s/temp_tags/tags/g; s/temp_badges/badges/g;' /media/seanleishman/Disk/stack_overflow_dump_1M/schema.sql

DROP TABLE IF EXISTS temp_posts;
DROP TABLE IF EXISTS temp_users;
DROP TABLE IF EXISTS temp_posthistory;
DROP TABLE IF EXISTS temp_comments;
DROP TABLE IF EXISTS temp_votes;
DROP TABLE IF EXISTS temp_postlinks;
DROP TABLE IF EXISTS temp_badges;

COMMIT;
