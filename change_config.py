from configparser import ConfigParser

config = ConfigParser()

config['db'] = {
    "host": '172.###.###.###',
    "port": "####",
    "user": 'cdmreader',
    "pwd": '#######',
    "schema_name": 'postgres'
}

config['ulcer_extract'] = {
    "query": "select * from cdm.note where note_id between 5000 and 7000 and note_text ilike '%%ulcer%%';",
    "out_schema": "public",
    "out_table": "ulcer_text_included_notes"
}

with open('./config/db_config.ini', 'w') as f:
    config.write(f)
