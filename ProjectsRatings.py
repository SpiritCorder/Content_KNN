# -*- coding: utf-8 -*-
"""
@author: sahan
"""

import os
import csv
import sys

from surprise import Dataset
from surprise import Reader

from collections import defaultdict


class ProjectsRatings:

    projectID_to_name = {}
    name_to_projectID = {}
    ratingsPath = './ratings.csv'
    projectsPath = './projects.csv'
    
    def loadProjectsRatingsData(self):

        # Look for files relative to the directory we are running from
        os.chdir(os.path.dirname(sys.argv[0]))

        ratingsDataset = 0
        self.projectID_to_name = {}
        self.name_to_projectID = {}

        reader = Reader(line_format='user rating item', sep=',', skip_lines=1)

        ratingsDataset = Dataset.load_from_file(self.ratingsPath, reader=reader)

        with open(self.projectsPath, newline='', encoding='ISO-8859-1') as csvfile:
                projectReader = csv.reader(csvfile)
                next(projectReader)  #Skip header line
                for row in projectReader:
                    projectID = int(row[0])
                    projectName = row[1]
                    self.projectID_to_name[projectID] = projectName
                    self.name_to_projectID[projectName] = projectID

        return ratingsDataset
    
    def getDomains(self):
        domains = defaultdict(list)
        domainIDs = {}
        maxDomainID = 0
        with open(self.projectsPath, newline='', encoding='ISO-8859-1') as csvfile:
            projectReader = csv.reader(csvfile)
            next(projectReader)  #Skip header line
            for row in projectReader:
                projectID = int(row[0])
                domainList = row[2].split('|')
                domainIDList = []
                for domain in domainList:
                    if domain in domainIDs:
                        domainID = domainIDs[domain]
                    else:
                        domainID = maxDomainID
                        domainIDs[domain] = domainID
                        maxDomainID += 1
                    domainIDList.append(domainID)
                domains[projectID] = domainIDList
        # Convert integer-encoded domain lists to bitfields that we can treat as vectors
        for (projectID, domainIDList) in domains.items():
            bitfield = [0] * maxDomainID
            for domainID in domainIDList:
                bitfield[domainID] = 1
            domains[projectID] = bitfield            
        
        return domains
    
    def getProjectName(self, projectID):
        if projectID in self.projectID_to_name:
            return self.projectID_to_name[projectID]
        else:
            return ""
    
    
