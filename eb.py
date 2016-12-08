# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 19:53:20 2016

@author: ONeill
"""
import math


#r1 is radius of primary (bigger) star
#r2 is radius of secondary (smaller) star
#b is impact parameter (projected distance b/w their centers)
#area of chord tangent to bottom of smaller star
#A = (r1^2/2)(theta-sintheta)
# theta = angle of circular segment

def getImpactParamTra(a, i, r1, e, argofperi):
    return (a*math.cos(i)/r1)*((1-e**2)/(1+e*math.sin(argofperi)))
    
def getImpactParamOcc(a, i, r1, e, argofperi):
    return (a*math.cos(i)/r1)*((1-e**2)/(1-e*math.sin(argofperi))) 


def primary_frac_area(r1,r2,b):
   x = math.sqrt(r1**2 - (b-r2)**2)
   coveredArea = ((r1**2)/2)*(2*math.arcsin(x/r1)-math.sin(2*math.arcsin(x/r1)))
   fracAreaOfCoveredPrimary = coveredArea / (math.pi*r1**2)
   return fracAreaOfCoveredPrimary