# CodeMe

This project was done as part of the course "Text Technologies for DS" in the University of Edinburgh. The core objective was to build a retrieval system designed to help developers
efficiently find relevant code snippets and discussions. Built on a version of the Stack Overflow dump, the system indexes millions of questions, answers, and code snippets, ensuring comprehensive coverage of real-world programming queries for users. Utilizing efficient, scalable and storage-efficient indexing techniques, domain-specific preprocessing techniques and various advanced retrieval and reranking methods the system enables fast and accurate code searches. It incorporates a modern backend powered by Flask for efficient query processing and ranking and has an intuitive userinterface, offering interactive features such as filtering. The link for the search engine can be provided upon request.  <br> The final report can be accessed here : [Final Report](https://github.com/annitziak/CodeMe/blob/main/final_report_group10.pdf)




## Enviornment Setup
Python v3.10.12 is used in the project. It is unclear if the code will work with other versions of Python.
The following instructions are for setting up the environment to run the code in this repository.
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

To ensure correct referencing of the modules in the repository, the following command should be run in the root directory of the repository.
```bash
pip install -e .
```

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
