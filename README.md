# Hybrid-search-RAG
Implementing hybrid search with RAG, using PGVector extension with PosgreSQL and PGAdmin4 tool

STEP 1:
PGVector Installation

    For Linux and Mac:
    Compile and install the extension (supports Postgres 13+)

        cd /tmp
        git clone --branch v0.8.0 https://github.com/pgvector/pgvector.git
        cd pgvector
        make
        make install # may need sudo



    Installation via Docker:

        docker pull pgvector/pgvector:pg17

    You can also build the image manually:

        git clone --branch v0.8.0 https://github.com/pgvector/pgvector.git
        cd pgvector
        docker build --pull --build-arg PG_MAJOR=17 -t myuser/pgvector .

STEP 2:
Install PostgreSQL

    $ docker run --name some-postgres -e POSTGRES_PASSWORD=mysecretpassword -p (port no.) -d postgres container name

Installation of pgAdmin4:

    setup reporsitory
    Install the public key for the repository (if not done previously):
    curl -fsS https://www.pgadmin.org/static/packages_pgadmin_org.pub | sudo gpg --dearmor -o /usr/share/keyrings/packages-pgadmin-org.gpg

    - Create the repository configuration file:
    sudo sh -c 'echo "deb [signed-by=/usr/share/keyrings/packages-pgadmin-org.gpg] https://ftp.postgresql.org/pub/pgadmin/pgadmin4/apt/$(lsb_release -cs) pgadmin4 main" > /etc/apt/sources.list.d/pgadmin4.list && apt update'

STEP 3:
- Install pgAdmin
# Install for both desktop and web modes:
* sudo apt install pgadmin4

# Install for desktop mode only:
* sudo apt install pgadmin4-desktop

# Install for web mode only: 
* sudo apt install pgadmin4-web

# Configure the webserver, if you installed pgadmin4-web:
* sudo /usr/pgadmin4/bin/setup-web.sh

STEP 4:
Setup pgAdmin4:

- Right click on servers
- Create -> server group
- Give server name then save
- Right click on you server 
- Register -> Server
- Give name (in general section)
- Host address = localhost ( Connection section)
- set port no., username, give password that you have given before while setting up PostgreSQL(POSTGRES_PASSWORD=mysecretpassword)
- Expand you serverin the left panel, then right click on Databases -> create -> database
- Give name to your database (e.g. vector_db), then save it
- Pip install psycopg2-binary pgvector into your system (terminal)
- from langchain.vectorstores.pgvector import PGVector
- Put your connection string into the code as ( Connection_string = "host=localhost dbname=(database_name) user=(user_name) password=(your_password) port=(port no.)"
- Right click on your database(vector_db) under Databases
- Click "Query Tool"
Enable the extension (do this once in each database where you want to use it)
- CREATE EXTENSION IF NOT EXISTS vector;
- CREATE TABLE annual_reports_index (
    chunk_index     numeric PRIMARY KEY,
    company_name    text,
    content         text,
    dense_embedding     vector(768),
    sparse_embedding       vector(1024)
);
- CREATE INDEX idx_sparse_vector
ON hybrid_sparse_chunks
USING ivfflat (sparse_vector vector_cosine_ops);

Use SQL commands in same query tool after encode the vectors(sparse and dense) and before retrieving the documents: 
- ALTER TABLE hybrid_sparse_chunks ADD COLUMN content_tsv tsvector;
- UPDATE hybrid_sparse_chunks SET content_tsv = to_tsvector('english', content);
- SELECT plainto_tsquery('english', 'your search text');
- SELECT * FROM hybrid_sparse_chunks
WHERE to_tsvector('english', content) @@ to_tsquery('water'); ( optional : To check the vectors are stored or not)

STEP 5:
- Create GroQ API KEY (Official Link : https://console.groq.com/keys) (choose model according to your choice)
- First, run the code step1_insert_to_db.py file to create vectors and store the embeddings into databse. (Will take much time, depending on the size of your data and hardware resources.)
- Finally, run the step2_hybrid_search.py file to run the streamlit application.

