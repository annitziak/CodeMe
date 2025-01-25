# Text Technologies for Data Science CW3

`potential_ideas.md` contains potential project ideas i.e. potential corpora to use and specific features of the system to be included

## Data Collection
A desciption of the StackOverfow data collection process is in `docs/stack_overflow.md`.
The scripts `scripts/stack_exchange.py` is used to read the raw XML files and use those to populate a PostgreSQL database.

> Note: The desciption of the database schema specifies certain foreign key constraints that are not actually enforced in the data itself. There is (WIP) to enfore these constraints in the database by adding dummy rows to the tables.

### Reading the Data
If a dump is available to you then you can use `scipts/read_dump.sql` to read the data into a PostgreSQL database.
Below are instructions to get setup with PostreSQl.

#### Setting up PostgreSQL
```bash
sudo apt-get install postgresql
```
Once a username and password is setup for the database, you can use the following commands to create the database and user.
```bash
sudo -u postgres psql
```
```sql
CREATE DATABASE stackoverflow;
```
OR
```bash
createdb stackoverflow
```

Hopefully the above works for you...

#### Reading the Data
```bash
cd scripts
chmod +x read_dump.bash
./read_dump.bash <path_to_dump> <database_name>
```
If the above does not work then it might have to do with the permissions of the database. You can try the following:
```bash
sudo -u postgres psql
```
```sql
GRANT ALL PRIVILEGES ON DATABASE stackoverflow TO <username>;
```
And if that does not work then you can try the following:
```bash
sudo -u postgres psql
```
```sql
ALTER DATABASE stackoverflow OWNER TO <username>;
```
If any of the above doesn't work put it in the chat.
