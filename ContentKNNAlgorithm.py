# -*- coding: utf-8 -*-
"""
@author: sahan
"""

from surprise import AlgoBase
from surprise import PredictionImpossible
from ProjectsRatings import ProjectsRatings
import math
import numpy as np
import heapq

class ContentKNNAlgorithm(AlgoBase):

    def __init__(self, k=10, sim_options={}):
        AlgoBase.__init__(self)
        self.k = k

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)


        pr = ProjectsRatings()
        domains = pr.getDomains()
        
        print("Computing content-based similarity matrix...")
            
        self.similarities = np.zeros((self.trainset.n_items, self.trainset.n_items))


        for thisRating in range(self.trainset.n_items):
            if (thisRating % 10 == 0):
                print(thisRating, " of ", self.trainset.n_items)
            for otherRating in range(thisRating+1, self.trainset.n_items):
                thisProjectID = int(self.trainset.to_raw_iid(thisRating))
                otherProjectID = int(self.trainset.to_raw_iid(otherRating))
                domainSimilarity = self.computeDomainSimilarity(thisProjectID, otherProjectID, domains)
                self.similarities[thisRating, otherRating] = domainSimilarity
                self.similarities[otherRating, thisRating] = self.similarities[thisRating, otherRating]

        print("...done.")

        return self
    
    def computeDomainSimilarity(self, project1, project2, domains):
        domains1 = domains[project1]
        domains2 = domains[project2]
        sumxx, sumxy, sumyy = 0, 0, 0
        for i in range(len(domains1)):
            x = domains1[i]
            y = domains2[i]
            sumxx += x * x
            sumyy += y * y
            sumxy += x * y
        
        return sumxy/math.sqrt(sumxx*sumyy)
    
    
    def estimate(self, u, i):
        
        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('User and/or item is unkown.')
        
    
        neighbors = []
        for rating in self.trainset.ur[u]:
            domainSimilarity = self.similarities[i,rating[0]]
            neighbors.append( (domainSimilarity, rating[1]) )
        
        k_neighbors = heapq.nlargest(self.k, neighbors, key=lambda t: t[0])
        
        simTotal = weightedSum = 0
        for (simScore, rating) in k_neighbors:
            if (simScore > 0):
                simTotal += simScore
                weightedSum += simScore * rating
            
        if (simTotal == 0):
            raise PredictionImpossible('No neighbors')

        predictedRating = weightedSum / simTotal

        return predictedRating
    