# -*- coding: utf-8 -*-
"""
@author: sahan
"""

from ProjectsRatings import ProjectsRatings
from Evaluator import Evaluator

from ContentKNNAlgorithm import ContentKNNAlgorithm

import random
import numpy as np


def LoadProjectsRatingsData():
    pr = ProjectsRatings()
    print("Loading project ratings...")
    data = pr.loadProjectsRatingsData()
    
    return (pr, data)

np.random.seed(0)
random.seed(0)


(pr, evaluationData) = LoadProjectsRatingsData()


evaluator = Evaluator(evaluationData)


contentKNN = ContentKNNAlgorithm()
evaluator.AddAlgorithm(contentKNN, "ContentKNN")


evaluator.SampleTopNRecs(pr)


