import psycopg2
import csv

class Etl:

    def __pg_connection(self, pg_str_conn):
    
        pg_conn = psycopg2.connect(pg_str_conn)

        return pg_conn

    def __pg_check_table_exists(self, pg_conn, schema, table):
        
        query = """
            SELECT max(1) as column FROM information_schema.tables
            WHERE table_schema = '{}'
            AND table_name = '{}';
        """.format(schema, table)

        pg_cursor = pg_conn.cursor()

        pg_cursor.execute(query)
        query_results = pg_cursor.fetchall()

        if query_results[0][0] == 1:
            return True
        else:
            return False

    def pg_load_from_csv_file(self, csv_source_file, file_delimiter, pg_str_conn, pg_schema, pg_dest_table):

        pg_conn = self.__pg_connection(pg_str_conn)

        table_exists = self.__pg_check_table_exists(pg_conn, pg_schema, pg_dest_table)
        print("Result check table exists: ", table_exists)

        if table_exists:

            pg_cursor = pg_conn.cursor()

            with open(csv_source_file, 'r') as f:

                
                next(f)
                pg_cursor.copy_from(f, pg_dest_table, sep=file_delimiter)
                
            pg_conn.commit()
                
                
                   
        pg_conn.close()

