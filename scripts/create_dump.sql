\set ON_ERROR_STOP on


CREATE TEMPORARY TABLE temp_posts AS
SELECT *
FROM posts
LIMIT 10000;
-- Could limit to some `creationdate` range

CREATE TEMPORARY TABLE temp_users AS
SELECT DISTINCT u.*
FROM users u
INNER JOIN temp_posts p ON p.owneruserid = u.id
UNION
SELECT DISTINCT u.*
FROM users u
INNER JOIN temp_posts p ON p.lasteditoruserid = u.id;

CREATE TEMPORARY TABLE temp_posthistory AS
SELECT ph.*
FROM posthistory ph
INNER JOIN temp_posts p ON ph.postid = p.id;

CREATE TEMPORARY TABLE temp_comments AS
SELECT c.*
FROM comments c
INNER JOIN temp_posts p ON c.postid = p.id;

CREATE TEMPORARY TABLE temp_votes AS
SELECT v.*
FROM votes v
INNER JOIN temp_posts p ON v.postid = p.id;

CREATE TEMPORARY TABLE temp_postlinks AS
SELECT pl.*
FROM postlinks pl
INNER JOIN temp_posts p ON pl.postid = p.id
   OR pl.relatedpostid = p.id;

\copy (SELECT * FROM temp_users ORDER BY id) TO output_path/users.csv WITH (FORMAT CSV, HEADER);

\copy (SELECT * FROM temp_posts ORDER BY id) TO output_path/posts.csv WITH (FORMAT CSV, HEADER);
\copy (SELECT * FROM temp_posthistory ORDER BY id) TO output_path/posthistory.csv WITH (FORMAT CSV, HEADER);
\copy (SELECT * FROM temp_comments ORDER BY id) TO output_path/comments.csv WITH (FORMAT CSV, HEADER);
\copy (SELECT * FROM temp_votes ORDER BY id) TO output_path/votes.csv WITH (FORMAT CSV, HEADER);
\copy (SELECT * FROM temp_postlinks ORDER BY id) TO output_path/postlinks.csv WITH (FORMAT CSV, HEADER);
\copy (SELECT * FROM tags ORDER BY id) TO output_path/tags.csv WITH (FORMAT CSV, HEADER);

\o 'output_path/schema.sql'
SELECT 'BEGIN;';

-- Users table
SELECT E'CREATE TABLE IF NOT EXISTS users (\n' ||
       E'    id INTEGER PRIMARY KEY\n' ||
       E'    reputation INTEGER,\n' ||
       E'    display_name TEXT,\n' ||
    E'creationdate TIMESTAMP,\n' ||
    E'lastaccessdate TIMESTAMP,\n' ||
    E'aboutme TEXT,\n' ||
    E'views INTEGER,\n' ||
    E'upvotes INTEGER,\n' ||
    E'downvotes INTEGER,\n' ||
       E');\n';

-- Posts table with custom constraints
SELECT E'CREATE TABLE IF NOT EXISTS posts (\n' ||
       E'    id INTEGER PRIMARY KEY,\n' ||
        E' posttypeid INTEGER,\n' ||
        E' acceptedanswerid INTEGER,\n' ||
        E' creationdate TIMESTAMP,\n' ||
        E' score INTEGER,\n' ||
        E' viewcount INTEGER,\n' ||
        E' body TEXT,\n' ||
       E'    owneruserid INTEGER REFERENCES users(id),\n' ||
       E'    lasteditoruserid INTEGER REFERENCES users(id),\n' ||
        E' lasteditordisplayname TEXT,\n' ||
        E'lasteditdate TIMESTAMP,\n' ||
        E' lastactivitydate TIMESTAMP,\n' ||
        E' title TEXT,\n' ||
        E' answercount INTEGER,\n' ||
        E' commentcount INTEGER,\n' ||
        E' favoritecount INTEGER,\n' ||
        E'contentlicense TEXT,\n' ||
       E'    CONSTRAINT valid_owner CHECK (\n' ||
       E'        owneruserid IN (SELECT id FROM temp_users)\n' ||
       E'    ),\n' ||
       E'    CONSTRAINT valid_editor CHECK (\n' ||
       E'        lasteditoruserid IS NULL OR\n' ||
       E'        lasteditoruserid IN (SELECT id FROM temp_users)\n' ||
       E'    )\n' ||
       E');\n';

-- PostHistory table
SELECT E'CREATE TABLE IF NOT EXISTS posthistory (\n' ||
       E'    id INTEGER PRIMARY KEY,\n' ||
    E' posthistorytypeid INTEGER,\n' ||
       E'    postid INTEGER REFERENCES posts(id),\n' ||
    E' revisionguid TEXT,\n' ||
    E' creationdate TIMESTAMP,\n' ||
       E'    userid INTEGER REFERENCES users(id)\n' ||
    E' text TEXT,\n' ||
    E' contentlicense TEXT,\n' ||
       E');\n';

-- Comments table
SELECT E'CREATE TABLE IF NOT EXISTS comments (\n' ||
       E'    id INTEGER PRIMARY KEY,\n' ||
       E'    postid INTEGER REFERENCES posts(id),\n' ||
    E' score INTEGER,\n' ||
    E' text TEXT,\n' ||
    E' creationdate TIMESTAMP,\n' ||
       E'    user_id INTEGER REFERENCES users(id)\n' ||
       E');\n';

SELECT E'CREATE TABLE IF NOT EXISTS votes (\n' ||
       E'    id INTEGER PRIMARY KEY,\n' ||
       E'    postid INTEGER REFERENCES posts(id),\n' ||
        E' votetypeid INTEGER,\n' ||
        E' creationdate TIMESTAMP,\n' ||
       E');\n';

SELECT E'CREATE TABLE IF NOT EXISTS postlinks (\n' ||
       E'    id INTEGER PRIMARY KEY,\n' ||
    E' creationdate TIMESTAMP,\n' ||
       E'    postid INTEGER REFERENCES posts(id),\n' ||
       E'    relatedpostid INTEGER REFERENCES posts(id),\n' ||
    E' linktype INTEGER,\n' ||
       E'    CONSTRAINT valid_related_post CHECK (\n' ||
       E'        relatedpostid IN (SELECT id FROM temp_posts)\n' ||
       E'    )\n' ||
       E');\n';

SELECT E'CREATE TABLE IF NOT EXISTS tags (\n' ||
       E'    id INTEGER PRIMARY KEY,\n' ||
       E'    tagname TEXT\n' ||
    E' count INTEGER,\n' ||
    E' excerptpostid INTEGER,\n' ||
    E' wikipostid INTEGER,\n' ||
       E');\n';

SELECT 'COMMIT;';
\o

-- Step 7: Clean up
DROP TABLE IF EXISTS temp_posts;
DROP TABLE IF EXISTS temp_users;
DROP TABLE IF EXISTS temp_posthistory;
DROP TABLE IF EXISTS temp_comments;
DROP TABLE IF EXISTS temp_votes;
DROP TABLE IF EXISTS temp_postlinks;
DROP TABLE IF EXISTS temp_post_tags;

-- Step 8: Commit transaction
COMMIT;
