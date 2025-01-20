# StackOverflow Data Dump Notes
The data dump contains the following files:
- `Badges.xml`
- `Comments.xml`
- `PostHistory.xml`
- `PostLinks.xml`
- `Posts.xml`
- `Tags.xml`
- `Users.xml`
- `Votes.xml`

The data dump is available at: https://archive.org/details/stackexchange and in total is 68GB compressed.
As such, we read the data in chunks and add it to a PostgreSQL database.

The schema of the database is as follows. The `id` column in each table is the primary key for that table.
Typically, `postId` references the `id` column in the `posts` table, `userId` references the `id` column in the `users` table, and `tagId` references the `id` column in the `tags` table.

     header     |                         table_md
----------------+-----------------------------------------------------------
 ## badges      | | Column | Type | Nullable | Default | Foreign Key       +
                | |--------|------|----------|----------|                  +
                | | id | integer | ✗ |  |                                  +
                | | userid | integer | ✓ |  |                              +
                | | name | text | ✓ |  |                                   +
                | | date | timestamp without time zone | ✓ |  |            +
                | | class | integer | ✓ |  |                               +
                | | tagbased | boolean | ✓ |  |
 ## comments    | | Column | Type | Nullable | Default | Foreign Key       +
                | |--------|------|----------|----------|                  +
                | | id | integer | ✗ |  |                                  +
                | | postid | integer | ✓ |  |                              +
                | | score | integer | ✓ |  |                               +
                | | text | text | ✓ |  |                                   +
                | | creationdate | timestamp without time zone | ✓ |  |    +
                | | userid | integer | ✓ |  | References (id) in users     +
 ## posthistory | | Column | Type | Nullable | Default | Foreign Key       +
                | |--------|------|----------|----------|                  +
                | | id | integer | ✗ |  |                                  +
                | | posthistorytypeid | integer | ✓ |  |                   +
                | | postid | integer | ✓ |  | Reference (id) in posts  +
                | | revisionguid | text | ✓ |  |                           +
                | | creationdate | timestamp without time zone | ✓ |  |    +
                | | userid | integer | ✓ |  | References (id) in users     +
                | | text | text | ✓ |  |                                   +
                | | contentlicense | text | ✓ |  |
 ## postlinks   | | Column | Type | Nullable | Default | Foreign Key       +
                | |--------|------|----------|----------|                  +
                | | id | integer | ✗ |  |                                  +
                | | creationdate | timestamp without time zone | ✓ |  |    +
                | | postid | integer | ✓ |  | Reference (id) in posts  +
                | | relatedpostid | integer | ✓ |  |                       +
                | | linktypeid | integer | ✓ |  |
 ## posts       | | Column | Type | Nullable | Default |                   +
                | |--------|------|----------|----------|                  +
                | | id | integer | ✗ |  |                                  +
                | | posttypeid | integer | ✓ |  |                          +
                | | acceptedanswerid | integer | ✓ |  |                    +
                | | creationdate | timestamp without time zone | ✓ |  |    +
                | | score | integer | ✓ |  |                               +
                | | viewcount | integer | ✓ |  |                           +
                | | body | text | ✓ |  |                                   +
                | | owneruserid | integer | ✓ |  | References (id) in users+
                | | ownerdisplayname | text | ✓ |  |                       +
                | | lasteditoruserid | integer | ✓ |  |                    +
                | | lasteditordisplayname | text | ✓ |  |                  +
                | | lasteditdate | timestamp without time zone | ✓ |  |    +
                | | lastactivitydate | timestamp without time zone | ✓ |  |+
                | | title | text | ✓ |  |                                  +
                | | tags | text | ✓ |  |                                   +
                | | answercount | integer | ✓ |  |                         +
                | | commentcount | integer | ✓ |  |                        +
                | | favoritecount | integer | ✓ |  |                       +
                | | contentlicense | text | ✓ |  |                         +
 ## tags        | | Column | Type | Nullable | Default |                   +
                | |--------|------|----------|----------|                  +
                | | id | integer | ✗ |  |                                  +
                | | tagname | text | ✓ |  |                                +
                | | count | integer | ✓ |  |                               +
                | | excerptpostid | integer | ✓ |  |                       +
                | | wikipostid | integer | ✓ |  |                          +
 ## users       | | Column | Type | Nullable | Default |                   +
                | |--------|------|----------|----------|                  +
                | | reputation | integer | ✓ |  |                          +
                | | creationdate | timestamp without time zone | ✓ |  |    +
                | | displayname | text | ✓ |  |                            +
                | | lastaccessdate | timestamp without time zone | ✓ |  |  +
                | | aboutme | text | ✓ |  |                                +
                | | views | integer | ✓ |  |                               +
                | | upvotes | integer | ✓ |  |                             +
                | | downvotes | integer | ✓ |  |                           +
                | | id | integer | ✗ |  |                                  +
 ## votes       | | Column | Type | Nullable | Default |                   +
                | |--------|------|----------|----------|                  +
                | | id | integer | ✗ |  |                                  +
                | | postid | integer | ✓ |  | References (id) in posts +
                | | votetypeid | integer | ✓ |  |                          +
                | | creationdate | timestamp without time zone | ✓ |  |    +
