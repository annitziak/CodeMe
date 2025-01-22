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

## Table Descriptions
### Posts
- `Id`: Id of the post
- `PostTypeId`: Type of the post (1 = Question, 2 = Answer, ..., 6=Moderator nomination, ... 17=CollectiveCollection)
- `AcceptedAnswerId`: Id of the accepted answer
- `CreationDate`: Date and time of creation
- `Score`: Score of the post (exists only for questions, answers and moderator nominations)
- `ViewCount`: Number of views (might not exist i.e. nullable)
- `Body`: The raw text of the post as rendered HTML
- `OwnerUserId`: Id of the owner (references `Users`)
- `OwnerDisplayName`: Display name of the owner
- `LastEditorUserId`: Id of the last editor (references `Users`)
- `lastActivityDate`: Date and time of last activity
- `Title`: Title of the post (question title for questions and tag name for tag wikis and excerpts)
- `Tags`: Space-separated list of tags
- `AnswerCount`: Number of answers (nullable)
- `CommentCount`: Number of comments (nullable)
- `FavoriteCount`: Number of favorites (nullable)
- Others are available and are listed [here](https://meta.stackexchange.com/questions/2677/database-schema-documentation-for-the-public-data-dump-and-sede/2678#2678)

All `PostTypeId` values are as follows:
- 1: Question
- 2: Answer
- 3: Orphaned Tag Wiki
- 4: TagWikiExcerpt
- 5: TagWiki
- 6: Moderator Nomination
- 7: Wiki Placeholder
- 8: Privilege Wiki
- 9: Article
- 10: HelpArticle
- 12: Collection
- 13: ModeratorQuestionnaireResponse
- 14: Announcement
- 15: CollectiveDiscussion
- 17: CollectiveCollection

### Users
- `Id`: Id of the user
- `Reputation`: Reputation of the user
- `CreationDate`: Date and time of creation
- `DisplayName`: Display name of the user
- `Views`: Number of views
- `UpVotes`: Number of upvotes
- `DownVotes`: Number of downvotes
- Others are available and are listed [here](https://meta.stackexchange.com/questions/2677/database-schema-documentation-for-the-public-data-dump-and-sede/2678#2678)

### Comments
- `Id`: Id of the comment
- `PostId`: Id of the post (references `Posts`)
- `Score`: Score of the comment
- `Text`: The raw text of the comment (comment body)
- `CreationDate`: Date and time of creation
- `UserId`: Id of the user (references `Users`)
- Others are available and are listed [here](https://meta.stackexchange.com/questions/2677/database-schema-documentation-for-the-public-data-dump-and-sede/2678#2678)

### Badges
- `Id`: Id of the badge
- `UserId`: Id of the user (references `Users`)
- `Name`: Name of the badge
- `Date`: Date and time of award
- `Class`: Class of the badge (1=Gold, 2=Silver, 3=Bronze)
- `TagBased`: Whether the badge is tag-based

### Tags
- `Id`: Id of the tag
- `TagName`: Name of the tag
- `Count`: Number of times the tag has been used
- `ExcerptPostId`: Id of the tag wiki excerpt (references `Posts`)
- `WikiPostId`: Id of the tag wiki (references `Posts`)
- Others are available and are listed [here](https://meta.stackexchange.com/questions/2677/database-schema-documentation-for-the-public-data-dump-and-sede/2678#2678)

### Votes
- `Id`: Id of the vote
- `PostId`: Id of the post (references `Posts`)
- `VoteTypeId`: Type of the vote (1=AcceptedByOriginator, 2=UpMod, 3=DownMod, 4=Offensive, 5=Favourite, 6=Close, 7=Reopen, 8=BountyStart, 9=BountyClose, 10=Deletion, 11=Undeletion, 12=Spam, 15=ModeratorReview, 16=ApproveEditSuggestion, [etc.](https://meta.stackexchange.com/questions/2677/database-schema-documentation-for-the-public-data-dump-and-sede/2678#2678))

### PostLinks
- `Id`: Id of the post link
- `CreationDate`: Date and time of creation
- `PostId`: Id of the post (references `Posts`)
- `RelatedPostId`: Id of the related post

### Other Tables
`PostHistory` is a major table that contains the history of posts.
However, the table is very large so it may be best to not handle it for the moment and base our work on `Posts`.
For live indexing, considering this table may be required if we are updating our database in real-time from older posts which have been updated.

A more detailed description of the fields and values can be found [here](https://meta.stackexchange.com/questions/2677/database-schema-documentation-for-the-public-data-dump-and-sede/2678#2678)

We read the data from the XML files and store it in a PostgreSQL database.
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
