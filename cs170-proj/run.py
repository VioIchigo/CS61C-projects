import numpy as np
import scipy as sp
import random
import networkx as nx
import matplotlib as plt
import sys
sys.path.append('..')
sys.path.append('../..')
import os
import argparse
import utils
import networkx as nx
import numpy as np
from student_utils_sp18 import *
from utils import *
from SetCoverV3 import *
from math import sqrt
from math import factorial
from itertools import combinations, permutations

import warnings
warnings.filterwarnings('ignore')
# for blabla in range(726):
blabla = 710
inp = "inputs/"+str(blabla)+".in"
data = read_file(inp)

number_of_kingdoms, list_of_kingdom_names, starting_kingdom, adjacency_matrix = data_parser(data)
n = number_of_kingdoms
INF  = 9999999999999999999999999999999999999999999

G = adjacency_matrix_to_graph(adjacency_matrix)

#         nx.draw(G, node_size=20, cmap=plt.cm.Blues,
#                     node_color=range(len(G)),
#                     prog='dot')
#         plt.show()

class City:
    def __init__(self, worldmap=None, id=None):
        self.worldmap = worldmap
        self.id = id

    def distanceTo(self, city):
        return self.worldmap[self.id][city.id]

    def __repr__(self):
        return str(self.id)


class TourManager:
    def __init__(self, IS = False):
        self.IS = IS
        self.destinationCities = []

    def addCity(self, city):
        self.destinationCities.append(city)

    def getCity(self, index):
        return self.destinationCities[index]

    def numberOfCities(self):
        return len(self.destinationCities)


class Tour:
    global adjacency_matrix
    global start
    def __init__(self,worldmap, tourmanager, tour=None):
        self.tourmanager = tourmanager
        self.worldmap = worldmap
        self.tour = []
        self.fitness = 0.0
        self.distance = 0
        if tour is not None:
            self.tour = tour
        else:
            for i in range(0, self.tourmanager.numberOfCities()):
                self.tour.append(None)

    def __len__(self):
        return len(self.tour)

    def __getitem__(self, index):
        return self.tour[index]

    def __setitem__(self, key, value):
        self.tour[key] = value

    def __repr__(self):
        geneString = "|"
        for i in range(0, self.tourSize()):
            geneString += str(self.getCity(i)) + "|"
        return geneString

    def generateIndividual(self):
        for cityIndex in range(0, self.tourmanager.numberOfCities()):
            self.setCity(cityIndex, self.tourmanager.getCity(cityIndex))
        random.shuffle(self.tour)

    def getCity(self, tourPosition):
        return self.tour[tourPosition]

    def setCity(self, tourPosition, city):
        self.tour[tourPosition] = city
        self.fitness = 0.0
        self.distance = 0

    def totalTime(self):
        totalTime = self.getDistance()
        for i in range(self.tourSize()):
            totalTime += adjacency_matrix[self.getCity(i).id][self.getCity(i).id]
        if self.tourmanager.IS:
            totalTime += adjacency_matrix[start][start]

        return totalTime

    def getFitness(self):
        if self.fitness == 0:
            self.fitness = self.totalTime()
        return self.fitness

    def getDistance(self):
        if self.distance == 0:
            tourDistance = 0
            for cityIndex in range(self.tourSize()-1):
                fromCity = self.getCity(cityIndex)
                destinationCity = self.getCity(cityIndex+1)
                tourDistance += fromCity.distanceTo(destinationCity)
            tourDistance += self.worldmap[start][self.tour[self.tourSize()-1].id]
            tourDistance += self.worldmap[start][self.tour[0].id]
            self.distance = tourDistance


        return self.distance

    def tourSize(self):
        return len(self.tour)


    def containsCity(self, city):
        return city in self.tour


class Population:
    def __init__(self, tourmanager, populationSize, initialise):
        self.tours = []
        for i in range(0, populationSize):
            self.tours.append(None)

        if initialise:
            for i in range(0, populationSize):
                newTour = Tour(distmatrix,tourmanager)
                newTour.generateIndividual()
                self.saveTour(i, newTour)

    def __setitem__(self, key, value):
        self.tours[key] = value

    def __getitem__(self, index):
        return self.tours[index]

    def saveTour(self, index, tour):
         self.tours[index] = tour

    def getTour(self, index):
        return self.tours[index]

    def getFittest(self):
        fittest = self.tours[0]
        for i in range(0, self.populationSize()):
            if fittest.getFitness() > self.getTour(i).getFitness():
                fittest = self.getTour(i)
        return fittest

    def populationSize(self):
        return len(self.tours)




class GA:
    def __init__(self, tourmanager):
        self.tourmanager = tourmanager
        self.mutationRate = 0.015
        self.tournamentSize = 5
        self.elitism = True

    def evolvePopulation(self, pop):
        newPopulation = Population(self.tourmanager, pop.populationSize(), False)
        elitismOffset = 0
        if self.elitism:
            newPopulation.saveTour(0, pop.getFittest())
            elitismOffset = 1

        for i in range(elitismOffset, newPopulation.populationSize()):
            parent1 = self.tournamentSelection(pop)
            parent2 = self.tournamentSelection(pop)
            child = self.crossover(parent1, parent2)
            newPopulation.saveTour(i, child)

        for i in range(elitismOffset, newPopulation.populationSize()):
            self.mutate(newPopulation.getTour(i))

        return newPopulation

    def crossover(self, parent1, parent2):
        child = Tour(distmatrix,self.tourmanager)

        startPos = int(random.random() * parent1.tourSize())
        endPos = int(random.random() * parent1.tourSize())

        for i in range(0, child.tourSize()):
            if startPos < endPos and i > startPos and i < endPos:
                child.setCity(i, parent1.getCity(i))
            elif startPos > endPos:
                if not (i < startPos and i > endPos):
                    child.setCity(i, parent1.getCity(i))

        for i in range(0, parent2.tourSize()):
            if not child.containsCity(parent2.getCity(i)):
                for ii in range(0, child.tourSize()):
                    if child.getCity(ii) == None:
                        child.setCity(ii, parent2.getCity(i))
                        break

        if start in child.tour:
            rec = child.tour[0]
            child.tour.remove(start)
            child.tour[0] = start
            child.tour.append(rec)

        return child

    def mutate(self, tour):
        for tourPos1 in range(0, tour.tourSize()):
            if random.random() < self.mutationRate:
                tourPos2 = int(tour.tourSize() * random.random())

                city1 = tour.getCity(tourPos1)
                city2 = tour.getCity(tourPos2)

                tour.setCity(tourPos2, city1)
                tour.setCity(tourPos1, city2)
        if start in tour:
            rec = tour[0]
            tour.remove(start)
            tour[0] = start
            tour.append(rec)

    def tournamentSelection(self, pop):
        tournament = Population(self.tourmanager, self.tournamentSize, False)
        for i in range(0, self.tournamentSize):
            randomId = int(random.random() * pop.populationSize())
            tournament.saveTour(i, pop.getTour(randomId))
        fittest = tournament.getFittest()
        return fittest



def toBinary(x):
    n = len(x)
    ret = [[0] * n for i in range(n)]
    for i in range(n):
        for j in range(n):
            if x[i][j] != 'x':
                   ret[i][j] = 1
    return ret

def toReal(x):
    INF  = 9999999999999999999999999999999999999999999
    n = len(x) # Get size of matrix
    ret = [[INF] * n for i in range(n)]
    for i in range(n):
        for j in range(n):
            if x[i][j] != 'x':
                   ret[i][j] = x[i][j]
    for i in range(n):
        ret[i][i] = 0
    return ret

binmatrix = toBinary(adjacency_matrix)
realmatrix = toReal(adjacency_matrix)
V = n
start = list_of_kingdom_names.index(starting_kingdom)

# Solves all pair shortest path via Floyd Warshall Algrorithm
def floydWarshall(graph):
    dist = list(map(lambda i : list(map(lambda j : j , i)) , graph))
    paths = [[str(list_of_kingdom_names[i])]*n for i in range(n)]
    for k in range(V):
        # pick all vertices as source one by one
        for i in range(V):
            # Pick all vertices as destination for the
            # above picked source
            for j in range(V):
                # If vertex k is on the shortest path from
                # i to j, then update the value of dist[i][j]
                if (dist[i][j] > dist[i][k]+ dist[k][j]):
                    dist[i][j] = dist[i][k]+ dist[k][j]
                    paths[i][j] = paths[i][k]+" "+paths[k][j]

    return dist,paths


if __name__ == '__main__':
    distmatrix, paths= floydWarshall(realmatrix)
    SC = True
    for i in range(n):
        for j in range(n-i):
            if adjacency_matrix[i][j]=='x':
                SC = False
                break
    if SC:
        optimal = adjacency_matrix[start][start]
        toconquer = start

        for i in range(n):
            if distmatrix[start][i] + distmatrix[i][start] + adjacency_matrix[i][i] < optimal:
                optimal = distmatrix[start][i] + distmatrix[i][start] + adjacency_matrix[i][i]
                toconquer = i
        paths[start][toconquer] += ' '
        paths[toconquer][start] += ' '
        if toconquer == start:
            realpath = [start]
        else:
            realpath = []
            rec = ''
            for i in paths[start][toconquer]:
                if i != ' ':
                    rec+=i
                else:
                    realpath.append(list_of_kingdom_names.index(rec))

            rec = ''
            for i in paths[toconquer][start]:
                if i != ' ':
                    rec+=i
                else:
                    realpath.append(list_of_kingdom_names.index(rec))
                    rec = ''
            realpath.append(start)
        orig_stdout = sys.stdout

        out = "outputs/"+str(blabla)+".out"
        f = open(out, 'w')
        sys.stdout = f
        print(' '.join(['{:1}'.format(list_of_kingdom_names[item]) for item in realpath]))
        print(list_of_kingdom_names[toconquer])
        sys.stdout = orig_stdout
        f.close()
    else:
        '''base case'''
        a = 1
        b = 0
        c = 0
        d = 1

        must_be_chosen, unchosen, covered, uncovered = select_leaves(adjacency_matrix)
        if len(covered) == len(adjacency_matrix):
            # print("case1")
            length = len(must_be_chosen)
            if length <= 7:
                possibilities = list(permutations(must_be_chosen, length))
                first_possibility = list(possibilities[0])
                shortest = distmatrix[start][first_possibility[0]]
                index = 0
                for i in range(length-1):
                    shortest += distmatrix[first_possibility[i]][first_possibility[i+1]]
                    shortest += adjacency_matrix[first_possibility[i]][first_possibility[i]]
                shortest += distmatrix[first_possibility[length-1]][start]
                shortest += adjacency_matrix[first_possibility[length-1]][first_possibility[length-1]]
                for i in range(1, factorial(length)):
                    chosen = list(possibilities[i])
                    new_distance = distmatrix[start][chosen[0]]
                    for j in range(length-1):
                        new_distance += distmatrix[chosen[j]][chosen[j+1]]
                        new_distance += adjacency_matrix[chosen[j]][chosen[j]]
                    new_distance += distmatrix[chosen[length-1]][start]
                    new_distance += adjacency_matrix[chosen[length-1]][chosen[length-1]]
                    if new_distance < shortest:
                        shortest = new_distance
                        index = i
                result = list(possibilities[index])

                if start in result:
                    tourmanager = TourManager(True)
                    for i in result:
                        if i != start:
                            city = City(distmatrix, i)
                            tourmanager1.addCity(city)
                else:
                    tourmanager = TourManager()
                    for i in result:
                        city = City(distmatrix, i)
                        tourmanager.addCity(city)
                soln = Tour(distmatrix, tourmanager)
                for cityIndex in range(0, tourmanager.numberOfCities()):
                    soln.setCity(cityIndex, soln.tourmanager.getCity(cityIndex))
                cost = soln.totalTime()
            else:
                if start in must_be_chosen:
                    tourmanager = TourManager(True)
                    for i in must_be_chosen:
                        if i != start:
                            city = City(distmatrix, i)
                            tourmanager.addCity(city)
                else:
                    tourmanager = TourManager()
                    for i in result:
                        city = City(distmatrix, i)
                        tourmanager.addCity(city)
                pop = Population(tourmanager, 2*(int(2*sqrt(len(adjacency_matrix)))), True)
                ga = GA(tourmanager)
                pop = ga.evolvePopulation(pop)
                for i in range(0, 3*len(adjacency_matrix)):
                    pop = ga.evolvePopulation(pop)
                soln = pop.getFittest()
                cost = soln.totalTime()
            orig_stdout = sys.stdout
            realpath=[]
            realpath += [paths[start][soln.tour[0].id]]
            for i in range(len(soln.tour)-1):
                realpath += [paths[soln.tour[i].id][soln.tour[i+1].id]]
            realpath += [paths[soln.tour[len(soln.tour)-1].id][start]]
            if soln.tourmanager.IS:
                record = []
                record.append(City(distmatrix, start))
                for city in soln.tour:
                    record.append(city)
                soln.tour = record
            if soln.tour[len(soln.tour)-1].id != start:
                realpath += [list_of_kingdom_names[start]]
            out = "outputs/"+str(blabla)+".out"
            f = open(out, 'w')
            sys.stdout = f
#                     print(soln1.totalTime(), soln2.totalTime())
            print(' '.join(['{:1}'.format(item) for item in realpath]))
            print(' '.join(['{:1}'.format(list_of_kingdom_names[int(item.id)]) for item in soln]))
            sys.stdout = orig_stdout
            f.close()
        else:
            # print("case2")
            must_be_chosen, unchosen, covered, uncovered = select_leaves(adjacency_matrix)
            result1 = setcoverV3(adjacency_matrix, must_be_chosen, unchosen, covered, uncovered, a, b)
            length1 = len(result1)
            if length1 <= 7:
                # print("factorial")
                possibilities = list(permutations(result1, length1))
                first_possibility = list(possibilities[0])
                shortest = distmatrix[start][first_possibility[0]]
                index = 0
                for i in range(length1-1):
                    shortest += distmatrix[first_possibility[i]][first_possibility[i+1]]
                    shortest += adjacency_matrix[first_possibility[i]][first_possibility[i]]
                shortest += distmatrix[first_possibility[length1-1]][start]
                shortest += adjacency_matrix[first_possibility[length1-1]][first_possibility[length1-1]]
                for i in range(1, factorial(length1)):
                    chosen = list(possibilities[i])
                    new_distance = distmatrix[start][chosen[0]]
                    for j in range(length1-1):
                        new_distance += distmatrix[chosen[j]][chosen[j+1]]
                        new_distance += adjacency_matrix[chosen[j]][chosen[j]]
                    new_distance += distmatrix[chosen[length1-1]][start]
                    new_distance += adjacency_matrix[chosen[length1-1]][chosen[length1-1]]
                    if new_distance < shortest:
                        shortest = new_distance
                        index = i
                result1 = list(possibilities[index])
                print(result1)

                if start in result1:
                    # print('start')
                    tourmanager1 = TourManager(True)
                    count = 0
                    for i in result1:
                        if i != start:
                            city = City(distmatrix, i)
                            tourmanager1.addCity(city)
                else:
                    tourmanager1 = TourManager()
                    for i in result1:
                        city = City(distmatrix, i)
                        tourmanager1.addCity(city)
                print(tourmanager1.numberOfCities())
                soln1 = Tour(distmatrix, tourmanager1)
                for i in tourmanager1.destinationCities:
                    print(i.id)
                for cityIndex in range(0, tourmanager1.numberOfCities()):
                    soln1.setCity(cityIndex, soln1.tourmanager.getCity(cityIndex))
                cost1 = soln1.totalTime()
                print(shortest, cost1)
                print(soln1.tour)
            else:
                # print("not factorial")
                if start in result1:
                    tourmanager1 = TourManager(True)
                    for i in result1:
                        if i != start:
                            city = City(distmatrix, i)
                            tourmanager1.addCity(city)
                else:
                    tourmanager1 = TourManager()
                    for i in result1:
                        city = City(distmatrix, i)
                        tourmanager1.addCity(city)
                pop1 = Population(tourmanager1, 2*(int((length1**2)/4)), True)
                ga1 = GA(tourmanager1)
                for i in range(0, 6*length1):
                    pop1 = ga1.evolvePopulation(pop1)
                soln1 = pop1.getFittest()
                cost1 = soln1.totalTime()



            must_be_chosen, unchosen, covered, uncovered = select_leaves(adjacency_matrix)
            result2 = setcoverV3(adjacency_matrix, must_be_chosen, unchosen, covered, uncovered, c, d)
            length2 = len(result2)
            if length2 <= 7:
                # print("factorial")
                possibilities = list(permutations(result2, length2))
                first_possibility = list(possibilities[0])
                shortest = distmatrix[start][first_possibility[0]]
                index = 0
                for i in range(length2-1):
                    shortest += distmatrix[first_possibility[i]][first_possibility[i+1]]
                    shortest += adjacency_matrix[first_possibility[i]][first_possibility[i]]
                shortest += distmatrix[first_possibility[length2-1]][start]
                shortest += adjacency_matrix[first_possibility[length2-1]][first_possibility[length2-1]]
                for i in range(1, factorial(length2)):
                    chosen = list(possibilities[i])
                    new_distance = distmatrix[start][chosen[0]]
                    for j in range(length2-1):
                        new_distance += distmatrix[chosen[j]][chosen[j+1]]
                        new_distance += adjacency_matrix[chosen[j]][chosen[j]]
                    new_distance += distmatrix[chosen[length2-1]][start]
                    new_distance += adjacency_matrix[chosen[length2-1]][chosen[length2-1]]
                    if new_distance < shortest:
                        shortest = new_distance
                        index = i
                result2 = list(possibilities[index])

                if start in result2:
                    tourmanager2 = TourManager(True)
                    for i in result2:
                        if i != start:
                            city = City(distmatrix, i)
                            tourmanager2.addCity(city)
                else:
                    tourmanager2 = TourManager()
                    for i in result2:
                        city = City(distmatrix, i)
                        tourmanager2.addCity(city)
                soln2 = Tour(distmatrix, tourmanager2)
                for cityIndex in range(0, tourmanager2.numberOfCities()):
                    soln2.setCity(cityIndex, soln2.tourmanager.getCity(cityIndex))
                cost2 = soln2.totalTime()
                print(soln2.tour)
                print(shortest, cost2)
            else:
                if start in result2:
                    tourmanager2 = TourManager(True)
                    for i in result2:
                        if i != start:
                            city = City(distmatrix, i)
                            tourmanager2.addCity(city)
                else:
                    tourmanager2 = TourManager()
                    for i in result2:
                        city = City(distmatrix, i)
                        tourmanager2.addCity(city)
                pop2 = Population(tourmanager2, 2*(int((length2**2)/4)), True)
                ga2 = GA(tourmanager2)
                for i in range(0, 6*length2):
                    pop2 = ga2.evolvePopulation(pop2)
                soln2 = pop2.getFittest()
                cost2 = soln2.totalTime()


            if cost1 < cost2:
                c = (a+c)/2
                d = (b+d)/2
                must_be_chosen, unchosen, covered, uncovered = select_leaves(adjacency_matrix)
                result2 = setcoverV3(adjacency_matrix, must_be_chosen, unchosen, covered, uncovered, c, d)
                length2 = len(result2)
                if length2 <= 7:
#                                 print('factorial')
                    possibilities = list(permutations(result2, length2))
                    first_possibility = list(possibilities[0])
                    shortest = distmatrix[start][first_possibility[0]]
                    index = 0
                    for i in range(length2-1):
                        shortest += distmatrix[first_possibility[i]][first_possibility[i+1]]
                        shortest += adjacency_matrix[first_possibility[i]][first_possibility[i]]
                    shortest += distmatrix[first_possibility[length2-1]][start]
                    shortest += adjacency_matrix[first_possibility[length2-1]][first_possibility[length2-1]]
                    for i in range(1, factorial(length2)):
                        chosen = list(possibilities[i])
                        new_distance = distmatrix[start][chosen[0]]
                        for j in range(length2-1):
                            new_distance += distmatrix[chosen[j]][chosen[j+1]]
                            new_distance += adjacency_matrix[chosen[j]][chosen[j]]
                        new_distance += distmatrix[chosen[length2-1]][start]
                        new_distance += adjacency_matrix[chosen[length2-1]][chosen[length2-1]]
                        if new_distance < shortest:
                            shortest = new_distance
                            index = i
                    result = list(possibilities[index])

                    if start in result:
                        tourmanager2 = TourManager(True)
                        for i in range(len(result)):
                            if i != start:
                                city = City(distmatrix, result[i])
                                tourmanager2.addCity(city)
                    else:
                        tourmanager2 = TourManager()
                        for i in range(len(result)):
                            city = City(distmatrix, result[i])
                            tourmanager2.addCity(city)
                    soln2 = Tour(distmatrix, tourmanager2)
                    for cityIndex in range(0, tourmanager2.numberOfCities()):
                        soln2.setCity(cityIndex, soln2.tourmanager.getCity(cityIndex))
                    cost2 = soln2.totalTime()
                else:
#                                 print('not factorial')
                    if start in result2:
                        tourmanager2 = TourManager(True)
                        for i in result2:
                            if i != start:
                                city = City(distmatrix, i)
                                tourmanager2.addCity(city)
                    else:
                        tourmanager2 = TourManager()
                        for i in result2:
                            city = City(distmatrix, i)
                            tourmanager2.addCity(city)
                    pop2 = Population(tourmanager2, 2*(int((length2**2)/4)), True)
                    ga2 = GA(tourmanager2)
                    for i in range(0, 6*length2):
                        pop2 = ga2.evolvePopulation(pop2)
                    soln2 = pop2.getFittest()
                    cost2 = soln2.totalTime()
            else:
                a = (a+c)/2
                b = (b+d)/2
                must_be_chosen, unchosen, covered, uncovered = select_leaves(adjacency_matrix)
                result1 = setcoverV3(adjacency_matrix, must_be_chosen, unchosen, covered, uncovered, a, b)
                length1 = len(result1)
                if length1 <= 7:
                    possibilities = list(permutations(result1, length1))
                    first_possibility = list(possibilities[0])
                    shortest = distmatrix[start][first_possibility[0]]
                    index = 0
                    for i in range(length1-1):
                        shortest += distmatrix[first_possibility[i]][first_possibility[i+1]]
                        shortest += adjacency_matrix[first_possibility[i]][first_possibility[i]]
                    shortest += distmatrix[first_possibility[length1-1]][start]
                    shortest += adjacency_matrix[first_possibility[length1-1]][first_possibility[length1-1]]
                    for i in range(1, factorial(length1)):
                        chosen = list(possibilities[i])
                        new_distance = distmatrix[start][chosen[0]]
                        for j in range(length1-1):
                            new_distance += distmatrix[chosen[j]][chosen[j+1]]
                            new_distance += adjacency_matrix[chosen[j]][chosen[j]]
                        new_distance += distmatrix[chosen[length1-1]][start]
                        new_distance += adjacency_matrix[chosen[length1-1]][chosen[length1-1]]
                        if new_distance < shortest:

                            shortest = new_distance
                            index = i
                    result = list(possibilities[index])
                    if start in result:
                        tourmanager1 = TourManager(True)
                        for i in result:
                            if i != start:
                                city = City(distmatrix, i)
                                tourmanager1.addCity(city)
                    else:
                        tourmanager1 = TourManager()
                        for i in result:
                            city = City(distmatrix, i)
                            tourmanager1.addCity(city)
                    soln1 = Tour(distmatrix, tourmanager1)
                    for cityIndex in range(0, tourmanager1.numberOfCities()):
                        soln1.setCity(cityIndex, soln1.tourmanager.getCity(cityIndex))
                    cost1 = soln1.totalTime()
                else:
                    if start in result1:
                        tourmanager1 = TourManager(True)
                        for i in result1:
                            if i != start:
                                city = City(distmatrix, i)
                                tourmanager1.addCity(city)
                    else:
                        tourmanager1 = TourManager()
                        for i in result1:
                            city = City(distmatrix, i)
                            tourmanager1.addCity(city)
                    pop1 = Population(tourmanager1, 2*(int((length1**2)/4)), True)
                    ga1 = GA(tourmanager1)
                    for i in range(0, 6*length1):
                        pop1 = ga1.evolvePopulation(pop1)
                    soln1 = pop1.getFittest()
                    cost1 = soln1.totalTime()
            if cost1 < cost2:
                c = (a+c)/2
                d = (b+d)/2
                must_be_chosen, unchosen, covered, uncovered = select_leaves(adjacency_matrix)
                result2 = setcoverV3(adjacency_matrix, must_be_chosen, unchosen, covered, uncovered, c, d)
                length2 = len(result2)
                if length2 <= 7:
                    possibilities = list(permutations(result2, length2))
                    first_possibility = list(possibilities[0])
                    shortest = distmatrix[start][first_possibility[0]]
                    index = 0
                    for i in range(length2-1):
                        shortest += distmatrix[first_possibility[i]][first_possibility[i+1]]
                        shortest += adjacency_matrix[first_possibility[i]][first_possibility[i]]
                    shortest += distmatrix[first_possibility[length2-1]][start]
                    shortest += adjacency_matrix[first_possibility[length2-1]][first_possibility[length2-1]]
                    for i in range(1, factorial(length2)):
                        chosen = list(possibilities[i])
                        new_distance = distmatrix[start][chosen[0]]
                        for j in range(length2-1):
                            new_distance += distmatrix[chosen[j]][chosen[j+1]]
                            new_distance += adjacency_matrix[chosen[j]][chosen[j]]
                        new_distance += distmatrix[chosen[length2-1]][start]
                        new_distance += adjacency_matrix[chosen[length2-1]][chosen[length2-1]]
                        if new_distance < shortest:
                            shortest = new_distance
                            index = i
                    result = list(possibilities[index])

                    if start in result:
                        tourmanager2 = TourManager(True)
                        for i in range(len(result)):
                            if i != start:
                                city = City(distmatrix, result[i])
                                tourmanager2.addCity(city)
                    else:
                        tourmanager2 = TourManager()
                        for i in range(len(result)):
                            city = City(distmatrix, result[i])
                            tourmanager2.addCity(city)
                    soln2 = Tour(distmatrix, tourmanager2)
                    for cityIndex in range(0, tourmanager2.numberOfCities()):
                        soln2.setCity(cityIndex, soln2.tourmanager.getCity(cityIndex))
                    cost2 = soln2.totalTime()
                else:
                    if start in result2:
                        tourmanager2 = TourManager(True)
                        for i in result2:
                            if i != start:
                                city = City(distmatrix, i)
                                tourmanager2.addCity(city)
                    else:
                        tourmanager2 = TourManager()
                        for i in result2:
                            city = City(distmatrix, i)
                            tourmanager2.addCity(city)
                    pop2 = Population(tourmanager2, 2*(int((length2**2)/4)), True)
                    ga2 = GA(tourmanager2)
                    for i in range(0, 6*length2):
                        pop2 = ga2.evolvePopulation(pop2)
                    soln2 = pop2.getFittest()
                    cost2 = soln2.totalTime()
            else:
                a = (a+c)/2
                b = (b+d)/2
                must_be_chosen, unchosen, covered, uncovered = select_leaves(adjacency_matrix)
                result1 = setcoverV3(adjacency_matrix, must_be_chosen, unchosen, covered, uncovered, a, b)
                length1 = len(result1)
                if length1 <= 7:
                    possibilities = list(permutations(result1, length1))
                    first_possibility = list(possibilities[0])
                    shortest = distmatrix[start][first_possibility[0]]
                    index = 0
                    for i in range(length1-1):
                        shortest += distmatrix[first_possibility[i]][first_possibility[i+1]]
                        shortest += adjacency_matrix[first_possibility[i]][first_possibility[i]]
                    shortest += distmatrix[first_possibility[length1-1]][start]
                    shortest += adjacency_matrix[first_possibility[length1-1]][first_possibility[length1-1]]
                    for i in range(1, factorial(length1)):
                        chosen = list(possibilities[i])
                        new_distance = distmatrix[start][chosen[0]]
                        for j in range(length1-1):
                            new_distance += distmatrix[chosen[j]][chosen[j+1]]
                            new_distance += adjacency_matrix[chosen[j]][chosen[j]]
                        new_distance += distmatrix[chosen[length1-1]][start]
                        new_distance += adjacency_matrix[chosen[length1-1]][chosen[length1-1]]
                        if new_distance < shortest:

                            shortest = new_distance
                            index = i
                    result = list(possibilities[index])
                    if start in result:
                        tourmanager1 = TourManager(True)
                        for i in result:
                            if i != start:
                                city = City(distmatrix, i)
                                tourmanager1.addCity(city)
                    else:
                        tourmanager1 = TourManager()
                        for i in result:
                            city = City(distmatrix, i)
                            tourmanager1.addCity(city)
                    soln1 = Tour(distmatrix, tourmanager1)
                    for cityIndex in range(0, tourmanager1.numberOfCities()):
                        soln1.setCity(cityIndex, soln1.tourmanager.getCity(cityIndex))
                    cost1 = soln1.totalTime()
                else:
                    if start in result1:
                        tourmanager1 = TourManager(True)
                        for i in result1:
                            if i != start:
                                city = City(distmatrix, i)
                                tourmanager1.addCity(city)
                    else:
                        tourmanager1 = TourManager()
                        for i in result1:
                            city = City(distmatrix, i)
                            tourmanager1.addCity(city)
                    pop1 = Population(tourmanager1, 2*(int((length1**2)/4)), True)
                    ga1 = GA(tourmanager1)
                    for i in range(0, 6*length1):
                        pop1 = ga1.evolvePopulation(pop1)
                    soln1 = pop1.getFittest()
                    cost1 = soln1.totalTime()
            if cost1 < cost2:
                c = (a+c)/2
                d = (b+d)/2
                must_be_chosen, unchosen, covered, uncovered = select_leaves(adjacency_matrix)
                result2 = setcoverV3(adjacency_matrix, must_be_chosen, unchosen, covered, uncovered, c, d)
                length2 = len(result2)
                if length2 <= 7:
                    possibilities = list(permutations(result2, length2))
                    first_possibility = list(possibilities[0])
                    shortest = distmatrix[start][first_possibility[0]]
                    index = 0
                    for i in range(length2-1):
                        shortest += distmatrix[first_possibility[i]][first_possibility[i+1]]
                        shortest += adjacency_matrix[first_possibility[i]][first_possibility[i]]
                    shortest += distmatrix[first_possibility[length2-1]][start]
                    shortest += adjacency_matrix[first_possibility[length2-1]][first_possibility[length2-1]]
                    for i in range(1, factorial(length2)):
                        chosen = list(possibilities[i])
                        new_distance = distmatrix[start][chosen[0]]
                        for j in range(length2-1):
                            new_distance += distmatrix[chosen[j]][chosen[j+1]]
                            new_distance += adjacency_matrix[chosen[j]][chosen[j]]
                        new_distance += distmatrix[chosen[length2-1]][start]
                        new_distance += adjacency_matrix[chosen[length2-1]][chosen[length2-1]]
                        if new_distance < shortest:
                            shortest = new_distance
                            index = i
                    result = list(possibilities[index])

                    if start in result:
                        tourmanager2 = TourManager(True)
                        for i in range(len(result)):
                            if i != start:
                                city = City(distmatrix, result[i])
                                tourmanager2.addCity(city)
                    else:
                        tourmanager2 = TourManager()
                        for i in range(len(result)):
                            city = City(distmatrix, result[i])
                            tourmanager2.addCity(city)
                    soln2 = Tour(distmatrix, tourmanager2)
                    for cityIndex in range(0, tourmanager2.numberOfCities()):
                        soln2.setCity(cityIndex, soln2.tourmanager.getCity(cityIndex))
                    cost2 = soln2.totalTime()
                else:
                    if start in result2:
                        tourmanager2 = TourManager(True)
                        for i in result2:
                            if i != start:
                                city = City(distmatrix, i)
                                tourmanager2.addCity(city)
                    else:
                        tourmanager2 = TourManager()
                        for i in result2:
                            city = City(distmatrix, i)
                            tourmanager2.addCity(city)
                    pop2 = Population(tourmanager2, 2*(int((length2**2)/4)), True)
                    ga2 = GA(tourmanager2)
                    for i in range(0, 6*length2):
                        pop2 = ga2.evolvePopulation(pop2)
                    soln2 = pop2.getFittest()
                    cost2 = soln2.totalTime()
            else:
                a = (a+c)/2
                b = (b+d)/2
                must_be_chosen, unchosen, covered, uncovered = select_leaves(adjacency_matrix)
                result1 = setcoverV3(adjacency_matrix, must_be_chosen, unchosen, covered, uncovered, a, b)
                length1 = len(result1)
                if length1 <= 7:
                    possibilities = list(permutations(result1, length1))
                    first_possibility = list(possibilities[0])
                    shortest = distmatrix[start][first_possibility[0]]
                    index = 0
                    for i in range(length1-1):
                        shortest += distmatrix[first_possibility[i]][first_possibility[i+1]]
                        shortest += adjacency_matrix[first_possibility[i]][first_possibility[i]]
                    shortest += distmatrix[first_possibility[length1-1]][start]
                    shortest += adjacency_matrix[first_possibility[length1-1]][first_possibility[length1-1]]
                    for i in range(1, factorial(length1)):
                        chosen = list(possibilities[i])
                        new_distance = distmatrix[start][chosen[0]]
                        for j in range(length1-1):
                            new_distance += distmatrix[chosen[j]][chosen[j+1]]
                            new_distance += adjacency_matrix[chosen[j]][chosen[j]]
                        new_distance += distmatrix[chosen[length1-1]][start]
                        new_distance += adjacency_matrix[chosen[length1-1]][chosen[length1-1]]
                        if new_distance < shortest:

                            shortest = new_distance
                            index = i
                    result = list(possibilities[index])
                    if start in result:
                        tourmanager1 = TourManager(True)
                        for i in result:
                            if i != start:
                                city = City(distmatrix, i)
                                tourmanager1.addCity(city)
                    else:
                        tourmanager1 = TourManager()
                        for i in result:
                            city = City(distmatrix, i)
                            tourmanager1.addCity(city)
                    soln1 = Tour(distmatrix, tourmanager1)
                    for cityIndex in range(0, tourmanager1.numberOfCities()):
                        soln1.setCity(cityIndex, soln1.tourmanager.getCity(cityIndex))
                    cost1 = soln1.totalTime()
                else:
                    if start in result1:
                        tourmanager1 = TourManager(True)
                        for i in result1:
                            if i != start:
                                city = City(distmatrix, i)
                                tourmanager1.addCity(city)
                    else:
                        tourmanager1 = TourManager()
                        for i in result1:
                            city = City(distmatrix, i)
                            tourmanager1.addCity(city)
                    pop1 = Population(tourmanager1, 2*(int((length1**2)/4)), True)
                    ga1 = GA(tourmanager1)
                    for i in range(0, 6*length1):
                        pop1 = ga1.evolvePopulation(pop1)
                    soln1 = pop1.getFittest()
                    cost1 = soln1.totalTime()
            if cost1 < cost2:
                c = (a+c)/2
                d = (b+d)/2
                must_be_chosen, unchosen, covered, uncovered = select_leaves(adjacency_matrix)
                result2 = setcoverV3(adjacency_matrix, must_be_chosen, unchosen, covered, uncovered, c, d)
                length2 = len(result2)
                if length2 <= 7:
                    possibilities = list(permutations(result2, length2))
                    first_possibility = list(possibilities[0])
                    shortest = distmatrix[start][first_possibility[0]]
                    index = 0
                    for i in range(length2-1):
                        shortest += distmatrix[first_possibility[i]][first_possibility[i+1]]
                        shortest += adjacency_matrix[first_possibility[i]][first_possibility[i]]
                    shortest += distmatrix[first_possibility[length2-1]][start]
                    shortest += adjacency_matrix[first_possibility[length2-1]][first_possibility[length2-1]]
                    for i in range(1, factorial(length2)):
                        chosen = list(possibilities[i])
                        new_distance = distmatrix[start][chosen[0]]
                        for j in range(length2-1):
                            new_distance += distmatrix[chosen[j]][chosen[j+1]]
                            new_distance += adjacency_matrix[chosen[j]][chosen[j]]
                        new_distance += distmatrix[chosen[length2-1]][start]
                        new_distance += adjacency_matrix[chosen[length2-1]][chosen[length2-1]]
                        if new_distance < shortest:
                            shortest = new_distance
                            index = i
                    result = list(possibilities[index])

                    if start in result:
                        tourmanager2 = TourManager(True)
                        for i in range(len(result)):
                            if i != start:
                                city = City(distmatrix, result[i])
                                tourmanager2.addCity(city)
                    else:
                        tourmanager2 = TourManager()
                        for i in range(len(result)):
                            city = City(distmatrix, result[i])
                            tourmanager2.addCity(city)
                    soln2 = Tour(distmatrix, tourmanager2)
                    for cityIndex in range(0, tourmanager2.numberOfCities()):
                        soln2.setCity(cityIndex, soln2.tourmanager.getCity(cityIndex))
                    cost2 = soln2.totalTime()
                else:
                    if start in result2:
                        tourmanager2 = TourManager(True)
                        for i in result2:
                            if i != start:
                                city = City(distmatrix, i)
                                tourmanager2.addCity(city)
                    else:
                        tourmanager2 = TourManager()
                        for i in result2:
                            city = City(distmatrix, i)
                            tourmanager2.addCity(city)
                    pop2 = Population(tourmanager2, 2*(int((length2**2)/4)), True)
                    ga2 = GA(tourmanager2)
                    for i in range(0, 6*length2):
                        pop2 = ga2.evolvePopulation(pop2)
                    soln2 = pop2.getFittest()
                    cost2 = soln2.totalTime()
            else:
                a = (a+c)/2
                b = (b+d)/2
                must_be_chosen, unchosen, covered, uncovered = select_leaves(adjacency_matrix)
                result1 = setcoverV3(adjacency_matrix, must_be_chosen, unchosen, covered, uncovered, a, b)
                length1 = len(result1)
                if length1 <= 7:
                    possibilities = list(permutations(result1, length1))
                    first_possibility = list(possibilities[0])
                    shortest = distmatrix[start][first_possibility[0]]
                    index = 0
                    for i in range(length1-1):
                        shortest += distmatrix[first_possibility[i]][first_possibility[i+1]]
                        shortest += adjacency_matrix[first_possibility[i]][first_possibility[i]]
                    shortest += distmatrix[first_possibility[length1-1]][start]
                    shortest += adjacency_matrix[first_possibility[length1-1]][first_possibility[length1-1]]
                    for i in range(1, factorial(length1)):
                        chosen = list(possibilities[i])
                        new_distance = distmatrix[start][chosen[0]]
                        for j in range(length1-1):
                            new_distance += distmatrix[chosen[j]][chosen[j+1]]
                            new_distance += adjacency_matrix[chosen[j]][chosen[j]]
                        new_distance += distmatrix[chosen[length1-1]][start]
                        new_distance += adjacency_matrix[chosen[length1-1]][chosen[length1-1]]
                        if new_distance < shortest:

                            shortest = new_distance
                            index = i
                    result = list(possibilities[index])
                    if start in result:
                        tourmanager1 = TourManager(True)
                        for i in result:
                            if i != start:
                                city = City(distmatrix, i)
                                tourmanager1.addCity(city)
                    else:
                        tourmanager1 = TourManager()
                        for i in result:
                            city = City(distmatrix, i)
                            tourmanager1.addCity(city)
                    soln1 = Tour(distmatrix, tourmanager1)
                    for cityIndex in range(0, tourmanager1.numberOfCities()):
                        soln1.setCity(cityIndex, soln1.tourmanager.getCity(cityIndex))
                    cost1 = soln1.totalTime()
                else:
                    if start in result1:
                        tourmanager1 = TourManager(True)
                        for i in result1:
                            if i != start:
                                city = City(distmatrix, i)
                                tourmanager1.addCity(city)
                    else:
                        tourmanager1 = TourManager()
                        for i in result1:
                            city = City(distmatrix, i)
                            tourmanager1.addCity(city)
                    pop1 = Population(tourmanager1, 2*(int((length1**2)/4)), True)
                    ga1 = GA(tourmanager1)
                    for i in range(0, 6*length1):
                        pop1 = ga1.evolvePopulation(pop1)
                    soln1 = pop1.getFittest()
                    cost1 = soln1.totalTime()
            if cost1 < cost2:
                c = (a+c)/2
                d = (b+d)/2
                must_be_chosen, unchosen, covered, uncovered = select_leaves(adjacency_matrix)
                result2 = setcoverV3(adjacency_matrix, must_be_chosen, unchosen, covered, uncovered, c, d)
                length2 = len(result2)
                if length2 <= 7:
                    possibilities = list(permutations(result2, length2))
                    first_possibility = list(possibilities[0])
                    shortest = distmatrix[start][first_possibility[0]]
                    index = 0
                    for i in range(length2-1):
                        shortest += distmatrix[first_possibility[i]][first_possibility[i+1]]
                        shortest += adjacency_matrix[first_possibility[i]][first_possibility[i]]
                    shortest += distmatrix[first_possibility[length2-1]][start]
                    shortest += adjacency_matrix[first_possibility[length2-1]][first_possibility[length2-1]]
                    for i in range(1, factorial(length2)):
                        chosen = list(possibilities[i])
                        new_distance = distmatrix[start][chosen[0]]
                        for j in range(length2-1):
                            new_distance += distmatrix[chosen[j]][chosen[j+1]]
                            new_distance += adjacency_matrix[chosen[j]][chosen[j]]
                        new_distance += distmatrix[chosen[length2-1]][start]
                        new_distance += adjacency_matrix[chosen[length2-1]][chosen[length2-1]]
                        if new_distance < shortest:
                            shortest = new_distance
                            index = i
                    result = list(possibilities[index])

                    if start in result:
                        tourmanager2 = TourManager(True)
                        for i in range(len(result)):
                            if i != start:
                                city = City(distmatrix, result[i])
                                tourmanager2.addCity(city)
                    else:
                        tourmanager2 = TourManager()
                        for i in range(len(result)):
                            city = City(distmatrix, result[i])
                            tourmanager2.addCity(city)
                    soln2 = Tour(distmatrix, tourmanager2)
                    for cityIndex in range(0, tourmanager2.numberOfCities()):
                        soln2.setCity(cityIndex, soln2.tourmanager.getCity(cityIndex))
                    cost2 = soln2.totalTime()
                else:
                    if start in result2:
                        tourmanager2 = TourManager(True)
                        for i in result2:
                            if i != start:
                                city = City(distmatrix, i)
                                tourmanager2.addCity(city)
                    else:
                        tourmanager2 = TourManager()
                        for i in result2:
                            city = City(distmatrix, i)
                            tourmanager2.addCity(city)
                    pop2 = Population(tourmanager2, 2*(int((length2**2)/4)), True)
                    ga2 = GA(tourmanager2)
                    for i in range(0, 6*length2):
                        pop2 = ga2.evolvePopulation(pop2)
                    soln2 = pop2.getFittest()
                    cost2 = soln2.totalTime()
            else:
                a = (a+c)/2
                b = (b+d)/2
                must_be_chosen, unchosen, covered, uncovered = select_leaves(adjacency_matrix)
                result1 = setcoverV3(adjacency_matrix, must_be_chosen, unchosen, covered, uncovered, a, b)
                length1 = len(result1)
                if length1 <= 7:
                    possibilities = list(permutations(result1, length1))
                    first_possibility = list(possibilities[0])
                    shortest = distmatrix[start][first_possibility[0]]
                    index = 0
                    for i in range(length1-1):
                        shortest += distmatrix[first_possibility[i]][first_possibility[i+1]]
                        shortest += adjacency_matrix[first_possibility[i]][first_possibility[i]]
                    shortest += distmatrix[first_possibility[length1-1]][start]
                    shortest += adjacency_matrix[first_possibility[length1-1]][first_possibility[length1-1]]
                    for i in range(1, factorial(length1)):
                        chosen = list(possibilities[i])
                        new_distance = distmatrix[start][chosen[0]]
                        for j in range(length1-1):
                            new_distance += distmatrix[chosen[j]][chosen[j+1]]
                            new_distance += adjacency_matrix[chosen[j]][chosen[j]]
                        new_distance += distmatrix[chosen[length1-1]][start]
                        new_distance += adjacency_matrix[chosen[length1-1]][chosen[length1-1]]
                        if new_distance < shortest:

                            shortest = new_distance
                            index = i
                    result = list(possibilities[index])
                    if start in result:
                        tourmanager1 = TourManager(True)
                        for i in result:
                            if i != start:
                                city = City(distmatrix, i)
                                tourmanager1.addCity(city)
                    else:
                        tourmanager1 = TourManager()
                        for i in result:
                            city = City(distmatrix, i)
                            tourmanager1.addCity(city)
                    soln1 = Tour(distmatrix, tourmanager1)
                    for cityIndex in range(0, tourmanager1.numberOfCities()):
                        soln1.setCity(cityIndex, soln1.tourmanager.getCity(cityIndex))
                    cost1 = soln1.totalTime()
                else:
                    if start in result1:
                        tourmanager1 = TourManager(True)
                        for i in result1:
                            if i != start:
                                city = City(distmatrix, i)
                                tourmanager1.addCity(city)
                    else:
                        tourmanager1 = TourManager()
                        for i in result1:
                            city = City(distmatrix, i)
                            tourmanager1.addCity(city)
                    pop1 = Population(tourmanager1, 2*(int((length1**2)/4)), True)
                    ga1 = GA(tourmanager1)
                    for i in range(0, 6*length1):
                        pop1 = ga1.evolvePopulation(pop1)
                    soln1 = pop1.getFittest()
                    cost1 = soln1.totalTime()
            if cost1 < cost2:
                c = (a+c)/2
                d = (b+d)/2
                must_be_chosen, unchosen, covered, uncovered = select_leaves(adjacency_matrix)
                result2 = setcoverV3(adjacency_matrix, must_be_chosen, unchosen, covered, uncovered, c, d)
                length2 = len(result2)
                if length2 <= 7:
                    possibilities = list(permutations(result2, length2))
                    first_possibility = list(possibilities[0])
                    shortest = distmatrix[start][first_possibility[0]]
                    index = 0
                    for i in range(length2-1):
                        shortest += distmatrix[first_possibility[i]][first_possibility[i+1]]
                        shortest += adjacency_matrix[first_possibility[i]][first_possibility[i]]
                    shortest += distmatrix[first_possibility[length2-1]][start]
                    shortest += adjacency_matrix[first_possibility[length2-1]][first_possibility[length2-1]]
                    for i in range(1, factorial(length2)):
                        chosen = list(possibilities[i])
                        new_distance = distmatrix[start][chosen[0]]
                        for j in range(length2-1):
                            new_distance += distmatrix[chosen[j]][chosen[j+1]]
                            new_distance += adjacency_matrix[chosen[j]][chosen[j]]
                        new_distance += distmatrix[chosen[length2-1]][start]
                        new_distance += adjacency_matrix[chosen[length2-1]][chosen[length2-1]]
                        if new_distance < shortest:
                            shortest = new_distance
                            index = i
                    result = list(possibilities[index])

                    if start in result:
                        tourmanager2 = TourManager(True)
                        for i in range(len(result)):
                            if i != start:
                                city = City(distmatrix, result[i])
                                tourmanager2.addCity(city)
                    else:
                        tourmanager2 = TourManager()
                        for i in range(len(result)):
                            city = City(distmatrix, result[i])
                            tourmanager2.addCity(city)
                    soln2 = Tour(distmatrix, tourmanager2)
                    for cityIndex in range(0, tourmanager2.numberOfCities()):
                        soln2.setCity(cityIndex, soln2.tourmanager.getCity(cityIndex))
                    cost2 = soln2.totalTime()
                else:
                    if start in result2:
                        tourmanager2 = TourManager(True)
                        for i in result2:
                            if i != start:
                                city = City(distmatrix, i)
                                tourmanager2.addCity(city)
                    else:
                        tourmanager2 = TourManager()
                        for i in result2:
                            city = City(distmatrix, i)
                            tourmanager2.addCity(city)
                    pop2 = Population(tourmanager2, 2*(int((length2**2)/4)), True)
                    ga2 = GA(tourmanager2)
                    for i in range(0, 6*length2):
                        pop2 = ga2.evolvePopulation(pop2)
                    soln2 = pop2.getFittest()
                    cost2 = soln2.totalTime()
            else:
                a = (a+c)/2
                b = (b+d)/2
                must_be_chosen, unchosen, covered, uncovered = select_leaves(adjacency_matrix)
                result1 = setcoverV3(adjacency_matrix, must_be_chosen, unchosen, covered, uncovered, a, b)
                length1 = len(result1)
                if length1 <= 7:
                    possibilities = list(permutations(result1, length1))
                    first_possibility = list(possibilities[0])
                    shortest = distmatrix[start][first_possibility[0]]
                    index = 0
                    for i in range(length1-1):
                        shortest += distmatrix[first_possibility[i]][first_possibility[i+1]]
                        shortest += adjacency_matrix[first_possibility[i]][first_possibility[i]]
                    shortest += distmatrix[first_possibility[length1-1]][start]
                    shortest += adjacency_matrix[first_possibility[length1-1]][first_possibility[length1-1]]
                    for i in range(1, factorial(length1)):
                        chosen = list(possibilities[i])
                        new_distance = distmatrix[start][chosen[0]]
                        for j in range(length1-1):
                            new_distance += distmatrix[chosen[j]][chosen[j+1]]
                            new_distance += adjacency_matrix[chosen[j]][chosen[j]]
                        new_distance += distmatrix[chosen[length1-1]][start]
                        new_distance += adjacency_matrix[chosen[length1-1]][chosen[length1-1]]
                        if new_distance < shortest:

                            shortest = new_distance
                            index = i
                    result = list(possibilities[index])
                    if start in result:
                        tourmanager1 = TourManager(True)
                        for i in result:
                            if i != start:
                                city = City(distmatrix, i)
                                tourmanager1.addCity(city)
                    else:
                        tourmanager1 = TourManager()
                        for i in result:
                            city = City(distmatrix, i)
                            tourmanager1.addCity(city)
                    soln1 = Tour(distmatrix, tourmanager1)
                    for cityIndex in range(0, tourmanager1.numberOfCities()):
                        soln1.setCity(cityIndex, soln1.tourmanager.getCity(cityIndex))
                    cost1 = soln1.totalTime()
                else:
                    if start in result1:
                        tourmanager1 = TourManager(True)
                        for i in result1:
                            if i != start:
                                city = City(distmatrix, i)
                                tourmanager1.addCity(city)
                    else:
                        tourmanager1 = TourManager()
                        for i in result1:
                            city = City(distmatrix, i)
                            tourmanager1.addCity(city)
                    pop1 = Population(tourmanager1, 2*(int((length1**2)/4)), True)
                    ga1 = GA(tourmanager1)
                    for i in range(0, 6*length1):
                        pop1 = ga1.evolvePopulation(pop1)
                    soln1 = pop1.getFittest()
                    cost1 = soln1.totalTime()
            if cost1 < cost2:
                c = (a+c)/2
                d = (b+d)/2
                must_be_chosen, unchosen, covered, uncovered = select_leaves(adjacency_matrix)
                result2 = setcoverV3(adjacency_matrix, must_be_chosen, unchosen, covered, uncovered, c, d)
                length2 = len(result2)
                if length2 <= 7:
                    possibilities = list(permutations(result2, length2))
                    first_possibility = list(possibilities[0])
                    shortest = distmatrix[start][first_possibility[0]]
                    index = 0
                    for i in range(length2-1):
                        shortest += distmatrix[first_possibility[i]][first_possibility[i+1]]
                        shortest += adjacency_matrix[first_possibility[i]][first_possibility[i]]
                    shortest += distmatrix[first_possibility[length2-1]][start]
                    shortest += adjacency_matrix[first_possibility[length2-1]][first_possibility[length2-1]]
                    for i in range(1, factorial(length2)):
                        chosen = list(possibilities[i])
                        new_distance = distmatrix[start][chosen[0]]
                        for j in range(length2-1):
                            new_distance += distmatrix[chosen[j]][chosen[j+1]]
                            new_distance += adjacency_matrix[chosen[j]][chosen[j]]
                        new_distance += distmatrix[chosen[length2-1]][start]
                        new_distance += adjacency_matrix[chosen[length2-1]][chosen[length2-1]]
                        if new_distance < shortest:
                            shortest = new_distance
                            index = i
                    result = list(possibilities[index])

                    if start in result:
                        tourmanager2 = TourManager(True)
                        for i in range(len(result)):
                            if i != start:
                                city = City(distmatrix, result[i])
                                tourmanager2.addCity(city)
                    else:
                        tourmanager2 = TourManager()
                        for i in range(len(result)):
                            city = City(distmatrix, result[i])
                            tourmanager2.addCity(city)
                    soln2 = Tour(distmatrix, tourmanager2)
                    for cityIndex in range(0, tourmanager2.numberOfCities()):
                        soln2.setCity(cityIndex, soln2.tourmanager.getCity(cityIndex))
                    cost2 = soln2.totalTime()
                else:
                    if start in result2:
                        tourmanager2 = TourManager(True)
                        for i in result2:
                            if i != start:
                                city = City(distmatrix, i)
                                tourmanager2.addCity(city)
                    else:
                        tourmanager2 = TourManager()
                        for i in result2:
                            city = City(distmatrix, i)
                            tourmanager2.addCity(city)
                    pop2 = Population(tourmanager2, 2*(int((length2**2)/4)), True)
                    ga2 = GA(tourmanager2)
                    for i in range(0, 6*length2):
                        pop2 = ga2.evolvePopulation(pop2)
                    soln2 = pop2.getFittest()
                    cost2 = soln2.totalTime()
            else:
                a = (a+c)/2
                b = (b+d)/2
                must_be_chosen, unchosen, covered, uncovered = select_leaves(adjacency_matrix)
                result1 = setcoverV3(adjacency_matrix, must_be_chosen, unchosen, covered, uncovered, a, b)
                length1 = len(result1)
                if length1 <= 7:
                    possibilities = list(permutations(result1, length1))
                    first_possibility = list(possibilities[0])
                    shortest = distmatrix[start][first_possibility[0]]
                    index = 0
                    for i in range(length1-1):
                        shortest += distmatrix[first_possibility[i]][first_possibility[i+1]]
                        shortest += adjacency_matrix[first_possibility[i]][first_possibility[i]]
                    shortest += distmatrix[first_possibility[length1-1]][start]
                    shortest += adjacency_matrix[first_possibility[length1-1]][first_possibility[length1-1]]
                    for i in range(1, factorial(length1)):
                        chosen = list(possibilities[i])
                        new_distance = distmatrix[start][chosen[0]]
                        for j in range(length1-1):
                            new_distance += distmatrix[chosen[j]][chosen[j+1]]
                            new_distance += adjacency_matrix[chosen[j]][chosen[j]]
                        new_distance += distmatrix[chosen[length1-1]][start]
                        new_distance += adjacency_matrix[chosen[length1-1]][chosen[length1-1]]
                        if new_distance < shortest:

                            shortest = new_distance
                            index = i
                    result = list(possibilities[index])
                    if start in result:
                        tourmanager1 = TourManager(True)
                        for i in result:
                            if i != start:
                                city = City(distmatrix, i)
                                tourmanager1.addCity(city)
                    else:
                        tourmanager1 = TourManager()
                        for i in result:
                            city = City(distmatrix, i)
                            tourmanager1.addCity(city)
                    soln1 = Tour(distmatrix, tourmanager1)
                    for cityIndex in range(0, tourmanager1.numberOfCities()):
                        soln1.setCity(cityIndex, soln1.tourmanager.getCity(cityIndex))
                    cost1 = soln1.totalTime()
                else:
                    if start in result1:
                        tourmanager1 = TourManager(True)
                        for i in result1:
                            if i != start:
                                city = City(distmatrix, i)
                                tourmanager1.addCity(city)
                    else:
                        tourmanager1 = TourManager()
                        for i in result1:
                            city = City(distmatrix, i)
                            tourmanager1.addCity(city)
                    pop1 = Population(tourmanager1, 2*(int((length1**2)/4)), True)
                    ga1 = GA(tourmanager1)
                    for i in range(0, 6*length1):
                        pop1 = ga1.evolvePopulation(pop1)
                    soln1 = pop1.getFittest()
                    cost1 = soln1.totalTime()
            if cost1 < cost2:
                c = (a+c)/2
                d = (b+d)/2
                must_be_chosen, unchosen, covered, uncovered = select_leaves(adjacency_matrix)
                result2 = setcoverV3(adjacency_matrix, must_be_chosen, unchosen, covered, uncovered, c, d)
                length2 = len(result2)
                if length2 <= 7:
                    possibilities = list(permutations(result2, length2))
                    first_possibility = list(possibilities[0])
                    shortest = distmatrix[start][first_possibility[0]]
                    index = 0
                    for i in range(length2-1):
                        shortest += distmatrix[first_possibility[i]][first_possibility[i+1]]
                        shortest += adjacency_matrix[first_possibility[i]][first_possibility[i]]
                    shortest += distmatrix[first_possibility[length2-1]][start]
                    shortest += adjacency_matrix[first_possibility[length2-1]][first_possibility[length2-1]]
                    for i in range(1, factorial(length2)):
                        chosen = list(possibilities[i])
                        new_distance = distmatrix[start][chosen[0]]
                        for j in range(length2-1):
                            new_distance += distmatrix[chosen[j]][chosen[j+1]]
                            new_distance += adjacency_matrix[chosen[j]][chosen[j]]
                        new_distance += distmatrix[chosen[length2-1]][start]
                        new_distance += adjacency_matrix[chosen[length2-1]][chosen[length2-1]]
                        if new_distance < shortest:
                            shortest = new_distance
                            index = i
                    result = list(possibilities[index])

                    if start in result:
                        tourmanager2 = TourManager(True)
                        for i in range(len(result)):
                            if i != start:
                                city = City(distmatrix, result[i])
                                tourmanager2.addCity(city)
                    else:
                        tourmanager2 = TourManager()
                        for i in range(len(result)):
                            city = City(distmatrix, result[i])
                            tourmanager2.addCity(city)
                    soln2 = Tour(distmatrix, tourmanager2)
                    for cityIndex in range(0, tourmanager2.numberOfCities()):
                        soln2.setCity(cityIndex, soln2.tourmanager.getCity(cityIndex))
                    cost2 = soln2.totalTime()
                else:
                    if start in result2:
                        tourmanager2 = TourManager(True)
                        for i in result2:
                            if i != start:
                                city = City(distmatrix, i)
                                tourmanager2.addCity(city)
                    else:
                        tourmanager2 = TourManager()
                        for i in result2:
                            city = City(distmatrix, i)
                            tourmanager2.addCity(city)
                    pop2 = Population(tourmanager2, 2*(int((length2**2)/4)), True)
                    ga2 = GA(tourmanager2)
                    for i in range(0, 6*length2):
                        pop2 = ga2.evolvePopulation(pop2)
                    soln2 = pop2.getFittest()
                    cost2 = soln2.totalTime()
            else:
                a = (a+c)/2
                b = (b+d)/2
                must_be_chosen, unchosen, covered, uncovered = select_leaves(adjacency_matrix)
                result1 = setcoverV3(adjacency_matrix, must_be_chosen, unchosen, covered, uncovered, a, b)
                length1 = len(result1)
                if length1 <= 7:
                    possibilities = list(permutations(result1, length1))
                    first_possibility = list(possibilities[0])
                    shortest = distmatrix[start][first_possibility[0]]
                    index = 0
                    for i in range(length1-1):
                        shortest += distmatrix[first_possibility[i]][first_possibility[i+1]]
                        shortest += adjacency_matrix[first_possibility[i]][first_possibility[i]]
                    shortest += distmatrix[first_possibility[length1-1]][start]
                    shortest += adjacency_matrix[first_possibility[length1-1]][first_possibility[length1-1]]
                    for i in range(1, factorial(length1)):
                        chosen = list(possibilities[i])
                        new_distance = distmatrix[start][chosen[0]]
                        for j in range(length1-1):
                            new_distance += distmatrix[chosen[j]][chosen[j+1]]
                            new_distance += adjacency_matrix[chosen[j]][chosen[j]]
                        new_distance += distmatrix[chosen[length1-1]][start]
                        new_distance += adjacency_matrix[chosen[length1-1]][chosen[length1-1]]
                        if new_distance < shortest:

                            shortest = new_distance
                            index = i
                    result = list(possibilities[index])
                    if start in result:
                        tourmanager1 = TourManager(True)
                        for i in result:
                            if i != start:
                                city = City(distmatrix, i)
                                tourmanager1.addCity(city)
                    else:
                        tourmanager1 = TourManager()
                        for i in result:
                            city = City(distmatrix, i)
                            tourmanager1.addCity(city)
                    soln1 = Tour(distmatrix, tourmanager1)
                    for cityIndex in range(0, tourmanager1.numberOfCities()):
                        soln1.setCity(cityIndex, soln1.tourmanager.getCity(cityIndex))
                    cost1 = soln1.totalTime()
                else:
                    if start in result1:
                        tourmanager1 = TourManager(True)
                        for i in result1:
                            if i != start:
                                city = City(distmatrix, i)
                                tourmanager1.addCity(city)
                    else:
                        tourmanager1 = TourManager()
                        for i in result1:
                            city = City(distmatrix, i)
                            tourmanager1.addCity(city)
                    pop1 = Population(tourmanager1, 2*(int((length1**2)/4)), True)
                    ga1 = GA(tourmanager1)
                    for i in range(0, 6*length1):
                        pop1 = ga1.evolvePopulation(pop1)
                    soln1 = pop1.getFittest()
                    cost1 = soln1.totalTime()
            if cost1 < cost2:
                c = (a+c)/2
                d = (b+d)/2
                must_be_chosen, unchosen, covered, uncovered = select_leaves(adjacency_matrix)
                result2 = setcoverV3(adjacency_matrix, must_be_chosen, unchosen, covered, uncovered, c, d)
                length2 = len(result2)
                if length2 <= 7:
                    possibilities = list(permutations(result2, length2))
                    first_possibility = list(possibilities[0])
                    shortest = distmatrix[start][first_possibility[0]]
                    index = 0
                    for i in range(length2-1):
                        shortest += distmatrix[first_possibility[i]][first_possibility[i+1]]
                        shortest += adjacency_matrix[first_possibility[i]][first_possibility[i]]
                    shortest += distmatrix[first_possibility[length2-1]][start]
                    shortest += adjacency_matrix[first_possibility[length2-1]][first_possibility[length2-1]]
                    for i in range(1, factorial(length2)):
                        chosen = list(possibilities[i])
                        new_distance = distmatrix[start][chosen[0]]
                        for j in range(length2-1):
                            new_distance += distmatrix[chosen[j]][chosen[j+1]]
                            new_distance += adjacency_matrix[chosen[j]][chosen[j]]
                        new_distance += distmatrix[chosen[length2-1]][start]
                        new_distance += adjacency_matrix[chosen[length2-1]][chosen[length2-1]]
                        if new_distance < shortest:
                            shortest = new_distance
                            index = i
                    result = list(possibilities[index])

                    if start in result:
                        tourmanager2 = TourManager(True)
                        for i in range(len(result)):
                            if i != start:
                                city = City(distmatrix, result[i])
                                tourmanager2.addCity(city)
                    else:
                        tourmanager2 = TourManager()
                        for i in range(len(result)):
                            city = City(distmatrix, result[i])
                            tourmanager2.addCity(city)
                    soln2 = Tour(distmatrix, tourmanager2)
                    for cityIndex in range(0, tourmanager2.numberOfCities()):
                        soln2.setCity(cityIndex, soln2.tourmanager.getCity(cityIndex))
                    cost2 = soln2.totalTime()
                else:
                    if start in result2:
                        tourmanager2 = TourManager(True)
                        for i in result2:
                            if i != start:
                                city = City(distmatrix, i)
                                tourmanager2.addCity(city)
                    else:
                        tourmanager2 = TourManager()
                        for i in result2:
                            city = City(distmatrix, i)
                            tourmanager2.addCity(city)
                    pop2 = Population(tourmanager2, 2*(int((length2**2)/4)), True)
                    ga2 = GA(tourmanager2)
                    for i in range(0, 6*length2):
                        pop2 = ga2.evolvePopulation(pop2)
                    soln2 = pop2.getFittest()
                    cost2 = soln2.totalTime()
            else:
                a = (a+c)/2
                b = (b+d)/2
                must_be_chosen, unchosen, covered, uncovered = select_leaves(adjacency_matrix)
                result1 = setcoverV3(adjacency_matrix, must_be_chosen, unchosen, covered, uncovered, a, b)
                length1 = len(result1)
                if length1 <= 7:
                    possibilities = list(permutations(result1, length1))
                    first_possibility = list(possibilities[0])
                    shortest = distmatrix[start][first_possibility[0]]
                    index = 0
                    for i in range(length1-1):
                        shortest += distmatrix[first_possibility[i]][first_possibility[i+1]]
                        shortest += adjacency_matrix[first_possibility[i]][first_possibility[i]]
                    shortest += distmatrix[first_possibility[length1-1]][start]
                    shortest += adjacency_matrix[first_possibility[length1-1]][first_possibility[length1-1]]
                    for i in range(1, factorial(length1)):
                        chosen = list(possibilities[i])
                        new_distance = distmatrix[start][chosen[0]]
                        for j in range(length1-1):
                            new_distance += distmatrix[chosen[j]][chosen[j+1]]
                            new_distance += adjacency_matrix[chosen[j]][chosen[j]]
                        new_distance += distmatrix[chosen[length1-1]][start]
                        new_distance += adjacency_matrix[chosen[length1-1]][chosen[length1-1]]
                        if new_distance < shortest:

                            shortest = new_distance
                            index = i
                    result = list(possibilities[index])
                    if start in result:
                        tourmanager1 = TourManager(True)
                        for i in result:
                            if i != start:
                                city = City(distmatrix, i)
                                tourmanager1.addCity(city)
                    else:
                        tourmanager1 = TourManager()
                        for i in result:
                            city = City(distmatrix, i)
                            tourmanager1.addCity(city)
                    soln1 = Tour(distmatrix, tourmanager1)
                    for cityIndex in range(0, tourmanager1.numberOfCities()):
                        soln1.setCity(cityIndex, soln1.tourmanager.getCity(cityIndex))
                    cost1 = soln1.totalTime()
                else:
                    if start in result1:
                        tourmanager1 = TourManager(True)
                        for i in result1:
                            if i != start:
                                city = City(distmatrix, i)
                                tourmanager1.addCity(city)
                    else:
                        tourmanager1 = TourManager()
                        for i in result1:
                            city = City(distmatrix, i)
                            tourmanager1.addCity(city)
                    pop1 = Population(tourmanager1, 2*(int((length1**2)/4)), True)
                    ga1 = GA(tourmanager1)
                    for i in range(0, 6*length1):
                        pop1 = ga1.evolvePopulation(pop1)
                    soln1 = pop1.getFittest()
                    cost1 = soln1.totalTime()

            if cost1 < cost2:
                soln = soln1
            else:
                soln = soln2
            orig_stdout = sys.stdout
            realpath=[]
            realpath += [paths[start][soln.tour[0].id]]
            for i in range(len(soln.tour)-1):
                realpath += [paths[soln.tour[i].id][soln.tour[i+1].id]]
            realpath += [paths[soln.tour[len(soln.tour)-1].id][start]]
            if soln.tourmanager.IS:
                record = []
                record.append(City(distmatrix, start))
                for city in soln.tour:
                    record.append(city)
                soln.tour = record
            if soln.tour[len(soln.tour)-1].id != start:
                realpath += [list_of_kingdom_names[start]]
            out = "outputs/"+str(blabla)+".out"
            f = open(out, 'w')
            sys.stdout = f
            print(soln1.totalTime(), soln2.totalTime())
            print(' '.join(['{:1}'.format(item) for item in realpath]))
            print(' '.join(['{:1}'.format(list_of_kingdom_names[int(item.id)]) for item in soln]))
            sys.stdout = orig_stdout
            f.close()
