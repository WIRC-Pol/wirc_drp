
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 14:14:00 2017

@author: ricky
Versioning

Get version strings
"""
import os

def get_version():
    '''
    This function returns the most recent version number tag reachable from a commit as string (e.g., 'v0.2.1'). If the tag points to the commit, then only the tag is shown. Otherwise, it suffixes the tag name with the number of additional commits on top of the tagged object and the abbreviated object name of the most recent commit (e.g., 'v0.2.1-14-g2414721', which is the 14th commit after the last update to the version number/tag). Use this function to get version name to insert into fits file header or wherever you want it.

    Inputs: 
        None

    Outputs: 
        vers -  version and commit number as string
    '''

    from git import Repo

    # repo = Repo(search_parent_directories=True)
    repo = Repo(os.environ["WIRCPOL_DRP"])

    #sha = repo.head.object.hexsha # Returns most recent git hash (commit ID) as string (e.g., '48dde70e3f472607ba6d72b40e98110479693a64'). Insert into file header or wherever you want it.

    #vers = repo.tags[-1].name # Returns most recent version number tag as string (e.g., 'v0.2.1'). Insert into file header or wherever you want it.

    vers = repo.git.describe()

    return vers


def get_hash():

    '''
    This function returns the mmost recent git hash (commit ID) as string (e.g., '48dde70e3f472607ba6d72b40e98110479693a64'). Insert into file header or wherever you want it.

    Inputs: 
        None

    Outputs: 
        sha -  version and commit number as string
    '''

    from git import Repo

    repo = Repo(search_parent_directories=True)

    sha = repo.head.object.hexsha

    return sha
