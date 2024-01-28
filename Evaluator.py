# -*- coding: utf-8 -*-
"""
@author: sahan
"""

from EvaluationData import EvaluationData
from EvaluatedAlgorithm import EvaluatedAlgorithm

class Evaluator:
    
    algorithms = []
    
    def __init__(self, dataset):
        ed = EvaluationData(dataset)
        self.dataset = ed
        
    def AddAlgorithm(self, algorithm, name):
        alg = EvaluatedAlgorithm(algorithm, name)
        self.algorithms.append(alg)
        
    def SampleTopNRecs(self, pr, testSubject=2, k=10):
        
        for algo in self.algorithms:
            print("\nUsing recommender ", algo.GetName())
            
            print("\nBuilding recommendation model...")
            trainSet = self.dataset.GetFullTrainSet()
        
            algo.GetAlgorithm().fit(trainSet)
            
            print("Computing recommendations...")
            testSet = self.dataset.GetAntiTestSetForUser(testSubject)
        

            predictions = algo.GetAlgorithm().test(testSet)
           
            recommendations = []
             
            print ("\nWe recommend projects:")
            for userID, movieID, actualRating, estimatedRating, _ in predictions:
                 intMovieID = int(movieID)
                 recommendations.append((intMovieID, estimatedRating))
             
            recommendations.sort(key=lambda x: x[1], reverse=True)
             
            for ratings in recommendations[:10]:
                 print(ratings[0], " : ",  pr.getProjectName(ratings[0]), ratings[1])
