# File-Database

This folder contains code related to adding fits headers to the database and will slowly 
contain more code for reducing raw data. 

Some of the code is under-tested or still untested. Moreover, we have not quite figured out the deployment
and testing architecture and are still waiting for riri to host the MySQL server (it is currently running
locally). Therefore, much of this code will not be testable to anyone not running a locally hosted MySQL 
server with the appropriately configured database.

In DB Setup:

The config_db.py file has the code to do the initial setup of the database. Make sure the master_BD.csv file is
in the same directory as config_db.py and that the user information is appropriate in the connection in the 
config_db.py file. Run this at a terminal, assuming you have a working MySQL server, and it should configure
the targets table, which is all that is necessary for much of the other code to be testable.


Requirements:

Python (written for 2.7)
astropy
numpy
wirc_drp
working MySQL server with WIRC_POL database

Usage:

Add files to raw_files table
$python read_to_rawdb.py YYYYMMDD [-f file1 file2 ...]
(Designed to be run in the same directory as directory with name YYYYMMDD)

More instructions will follow....
