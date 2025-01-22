import xml.etree.ElementTree as ET
import logging
import datetime
import pprint

from concurrent.futures import ThreadPoolExecutor
from utils.db_connection import DBConnection

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(message)s", datefmt="%d-%b-%y %H:%M:%S"
)
logger = logging.getLogger(__name__)


class StackExchangeImporter:
    def __init__(self, db_params):
        self.db_connection = DBConnection(db_params).connect()

        self.primary_keys = {}
        self.foreign_keys = {}

    def __del__(self):
        self.db_connection.disconnect()

    def add_primary_key(self, table_name, column_name):
        self.primary_keys[table_name] = column_name

    def add_foreign_key(self, table_name, column_name, ref_table, ref_column):
        if table_name not in self.foreign_keys:
            self.foreign_keys[table_name] = []

        self.foreign_keys[table_name].append((column_name, ref_table, ref_column))

    def create_table_from_xml(self, xml_file):
        context = ET.iterparse(xml_file, events=("start", "end"))
        table_name = os.path.basename(xml_file).split(".")[0]
        columns = []

        if False:
            self.db_connection.execute(f"DROP TABLE IF EXISTS {table_name}")

        pk_col = self.primary_keys.get(table_name.lower(), None)

        for _, elem in context:
            if elem.tag == "row":
                for key, value in elem.attrib.items():
                    if value.isdigit():
                        columns.append(f"{key} INT")
                    elif value.lower() in ["true", "false"]:
                        columns.append(f"{key} BOOLEAN")
                    elif self._is_timestamp(value):
                        columns.append(f"{key} TIMESTAMP")
                    elif value[0] == "-" and value[1:].isdigit():
                        columns.append(f"{key} INT")
                    else:
                        columns.append(f"{key} TEXT")
                break

        if pk_col is not None:
            columns.append(f"PRIMARY KEY ({pk_col})")

        if False:
            create_table_query = f"CREATE TABLE {table_name} ({', '.join(columns)})"
            self.db_connection.execute(create_table_query, commit=True)
            logger.info(
                f"Table {table_name} created with statement: '{pprint.pformat(create_table_query)}'"
            )

    def insert_data_from_xml(self, xml_file):
        try:
            context = ET.iterparse(xml_file, events=("start", "end"))
            table_name = os.path.basename(xml_file).split(".")[0]

            batch_size = 1000
            batch = []
            columns = None
            pk_col = self.primary_keys.get(table_name.lower(), None)

            for _, elem in context:
                if elem.tag == "row":
                    if columns is None:
                        columns = elem.attrib.keys()

                    if pk_col is not None:
                        pk_val = elem.attrib.get(pk_col)
                        if pk_val is None or pk_val.strip() == "":
                            logger.debug(
                                f"Skipping row without primary key value in {table_name} {elem.attrib.items()}"
                            )
                            continue

                    row = [elem.attrib.get(column) for column in columns]
                    batch.append(row)

                    if len(batch) >= batch_size:
                        self._insert_batch(table_name, columns, batch)
                        batch = []

                    elem.clear()

            if batch:
                self._insert_batch(table_name, columns, batch)

            logger.info(f"Data from {xml_file} inserted into {table_name}")

        except Exception as e:
            self.db_connection.rollback()
            logger.error(f"Error inserting data from {xml_file}: {e}")

    def alter_table(self, table_name):
        logger.info(f"Altering table {table_name}")
        if table_name.lower() in self.foreign_keys:
            foreign_keys = self.foreign_keys[table_name.lower()]
            for column_name, ref_table, ref_column in foreign_keys:
                drop_constraint_query = f"ALTER TABLE {table_name} DROP CONSTRAINT IF EXISTS {table_name}_{column_name}_fkey"
                logger.info(
                    f"Dropping constraint with statement: '{pprint.pformat(drop_constraint_query)}'"
                )
                self.db_connection.execute(drop_constraint_query, commit=False)
                logger.info(f"Constraint dropped on table {table_name}")

                index_create_quer = f"CREATE INDEX ON {table_name} ({column_name})"
                logger.info(
                    f"Creating index with statement: '{pprint.pformat(index_create_quer)}'"
                )
                self.db_connection.execute(index_create_quer, commit=True)

                try:
                    alter_table_query = f"ALTER TABLE {table_name} ADD FOREIGN KEY ({column_name}) REFERENCES {ref_table} ({ref_column})"
                    logger.info(
                        f"Altering table {table_name} with statement: '{pprint.pformat(alter_table_query)}'"
                    )
                    self.db_connection.execute(alter_table_query, commit=True)
                    logger.info(f"Table {table_name} altered")
                except Exception as e:
                    logger.error(f"Error altering table {table_name}: {e}")

    def _insert_batch(self, table_name, columns, batch):
        insert_query = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES %s ON CONFLICT DO NOTHING"
        self.db_connection.execute_values(
            insert_query, batch, page_size=len(batch), commit=True
        )

    def _is_timestamp(self, value):
        try:
            datetime.datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%f")
            return True
        except ValueError:
            return False

    def process_xml(self, xml_file):
        self.create_table_from_xml(xml_file)
        self.insert_data_from_xml(xml_file)

    def process_directory(self, xml_dir, num_workers, process_xml=True):
        xml_files = [
            os.path.join(xml_dir, f) for f in os.listdir(xml_dir) if f.endswith(".xml")
        ]

        if process_xml:
            with ThreadPoolExecutor(
                max_workers=min(len(xml_files), num_workers)
            ) as executor:
                futures = [executor.submit(self.process_xml, f) for f in xml_files]

                for future in futures:
                    future.result()

        logger.info(
            f"All XML files processed. Altering tables with foreign keys {self.foreign_keys}"
        )
        for table_name in self.foreign_keys.keys():
            self.alter_table(table_name)

    def _dict_to_xml_attr(self, d):
        return " ".join([f'{k}="{v}"' for k, v in d.items()])

    def create_test_xml(self, xml_file, num_rows=1000, save_dir=None):
        table_name = os.path.basename(xml_file).split(".")[0]

        if (
            save_dir is None
            or not os.path.exists(save_dir)
            or not os.path.isdir(save_dir)
        ):
            raise Exception(f"Directory {save_dir} does not exist")

        test_xml_file = os.path.join(save_dir, f"{table_name}_test.xml")

        context = ET.iterparse(xml_file, events=("start", "end"))
        root = None
        columns = None
        count = 0

        with open(test_xml_file, "w") as f:
            f.write('<?xml version="1.0" encoding="utf-8"?>\n')
            f.write(f"<{table_name}>\n")

            for _, elem in context:
                if elem.tag == "row":
                    if columns is None:
                        columns = elem.attrib.keys()

                    if root is None:
                        root = elem.tag

                    row = [elem.attrib.get(column) for column in columns]
                    f.write(
                        f"  <row {self._dict_to_xml_attr(dict(zip(columns, row)))}/>\n"
                    )
                    count += 1

                    if count >= num_rows:
                        break

                    elem.clear()

            f.write(f"</{table_name}>")

        logger.info(f"Test XML file {test_xml_file} created")

    def process_directory_create_test_xml(self, xml_dir, save_dir):
        xml_files = [
            os.path.join(xml_dir, f) for f in os.listdir(xml_dir) if f.endswith(".xml")
        ]

        for xml_file in xml_files:
            self.create_test_xml(xml_file, save_dir=save_dir)


if __name__ == "__main__":
    import os

    db_params = {
        "dbname": os.getenv("POSTGRES_DB"),
        "user": os.getenv("POSTGRES_USER"),
    }

    importer = StackExchangeImporter(db_params)

    import argparse
    import os

    parser = argparse.ArgumentParser(
        description="Utility script to read Stack Exchange XML files and import them into a PostgreSQL database"
    )
    parser.add_argument("--xml-file", help="Path to the XML file")
    parser.add_argument(
        "--xml-dir",
        help="Path to the directory of XML files. If not set we will use STACKOVERFLOW_XML_DIR environment variable",
    )

    parser.add_argument(
        "--save-dir", help="Path to the directory to save test XML files"
    )
    parser.add_argument(
        "--create-test-xml",
        action="store_true",
        help="Create test XML files from the generated table",
    )
    parser.add_argument(
        "--alter-table",
        action="store_true",
        help="Alter the table to add foreign keys",
    )

    args = parser.parse_args()

    importer.add_primary_key("posthistory", "Id")
    importer.add_primary_key("posts", "Id")
    importer.add_primary_key("users", "Id")
    importer.add_primary_key("votes", "Id")
    importer.add_primary_key("comments", "Id")
    importer.add_primary_key("badges", "Id")
    importer.add_primary_key("tags", "Id")
    importer.add_primary_key("postlinks", "Id")

    importer.add_foreign_key("posthistory", "postId", "posts", "Id")
    importer.add_foreign_key("posthistory", "userId", "users", "Id")

    importer.add_foreign_key("posts", "owneruserId", "users", "Id")
    importer.add_foreign_key("posts", "lasteditoruserId", "users", "Id")

    importer.add_foreign_key("comments", "userId", "users", "Id")
    importer.add_foreign_key("comments", "postId", "posts", "Id")

    importer.add_foreign_key("votes", "postId", "posts", "Id")

    importer.add_foreign_key("badges", "userId", "users", "Id")

    importer.add_foreign_key("postlinks", "postId", "posts", "Id")
    importer.add_foreign_key("postlinks", "relatedpostId", "posts", "Id")

    importer.add_foreign_key("tags", "excerptpostId", "posts", "Id")

    try:
        if args.xml_file:
            importer.process_xml(args.xml_file)
            exit(0)

        if args.xml_dir is None:
            args.xml_dir = os.getenv("STACKOVERFLOW_XML_DIR")
            logger.info(
                f"No XML directory provided. Using environment variable {args.xml_dir}"
            )

        if args.xml_dir:
            if not args.create_test_xml:
                importer.process_directory(
                    args.xml_dir, num_workers=4, process_xml=not args.alter_table
                )
            else:
                importer.process_directory_create_test_xml(args.xml_dir, args.save_dir)

    except Exception as e:
        logger.error(e)

    del importer
