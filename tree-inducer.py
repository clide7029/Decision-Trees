import sys
import numpy as np
import pandas as pd

data = pd.read_table("Decision-Tree/voting-data.tsv", names=["rep","party","voteRecord"], header=None)
# data = pd.read_table(sys.argv[1], names=["rep","party","voteRecord"], header=None)

print(data.voteRecord[0][0])

#entropy(s) = -p(A)logp(A) -p(B)logp(B) || 
# -sum(p(s)logp(s)) where s <= states

#gain(s) = entropy(s) - sum


class Node:
    def __init__(self, data, splitList=[], depth=0, parent=None, decision=None, maxDepth=20):
        self.data = data
        self.splitList = splitList
        self.depth = depth
        self.parent = parent
        self.decision = decision
        self.maxDepth = maxDepth
        self.best = {}
        self.splitIssue = -1
        self.isLeaf = False
        self.classify = None
        self.classified = None
        self.demRatio = 0
        self.yayNode = None
        self.nayNode = None
        self.absNode = None

    def entropy(self, repList, split=''):
        # for i in range(len(self.data.voteRecord[0])):
        #     for issue in self.data.voteRecord[i]:
        entropy = 0
        demCount = 0
        repCount = 0
        # print(repList)
        if len(repList.index) == 0 and self.parent is not None:
            self.classify, self.classified = self.classifyNode()
        else:
            for party in repList["party"]:
                # print('rep ',rep,'issue ',issue)
                
                if party == "D":
                    demCount += 1
                elif party == "R":
                    repCount += 1
        # print("demCount ", demCount)
        # print("repCount ", repCount)
            entropy = 1
            if demCount != 0 and repCount != 0:
                demRatio = demCount / (demCount + repCount)
                repRatio = repCount / (demCount + repCount)
                
                if demCount == repCount:
                    # print('counts tied need parent classify')
                    self.classify, self.classified = self.classifyNode()
                elif split == 'self':
                    self.classified = ("self", self.depth, 'D={0}, R={1}, reps={2}'.format(demCount, repCount, len(repList)))
                    if demCount > repCount:
                        self.classify = 'D'
                    elif repCount > demCount:
                        self.classify = 'R'
                # print(demRatio)
                # print(repRatio)

                entropy = -demRatio*np.log2(demRatio) - repRatio*np.log2(repRatio)
            else:
                entropy = 0
                if split == "self":
                    self.classified = ("self", self.depth, 'replist = {0}, D={1}, R={2}'.format(len(repList), demCount, repCount))
                    if demCount == 0:
                        self.classify = "R"
                    elif repCount == 0:
                        self.classify = "D"
                
        
        
        # print("issue ", i, " has entropy = ", entropy)
        # print("demCount = ",demCount,"repCount = ",repCount)
        return entropy

    def classifyNode(self):
        repList = self.parent.data.copy()
        demCount = 0
        repCount = 0
        res = ""
        for party in repList["party"]:
            # print('rep ',rep,'issue ',issue)

            if party == "D":
                demCount += 1
            elif party == "R":
                repCount += 1
                
        if demCount == repCount:
            res, classified = self.parent.classifyNode()
        else:
            classified = ("parent", self.parent.depth, 'D={0}, R={1}, reps={2}'.format(demCount, repCount, repCount+demCount))
            if demCount > repCount:
                res = "D"
            elif repCount > demCount:
                res = "R"
        
        # print('classified by parent as ', res)
        
        return res, classified
            
    def calcInfoGain(self, currentEntropy):
        best = {'val': 0, 'idx': -1, 'yay': pd.DataFrame(), 'nay': pd.DataFrame(), 'abs': pd.DataFrame()}
        # print(currentEntropy)
        # print("inside calc gain")
        # print(self.data.head(1).voteRecord.str.toString())
        # print(self.data.loc[self.data.index, 'voteRecord'].iat[0])
        # print(self.data.to_string())
        for i in range(len(self.data.loc[self.data.index, 'voteRecord'].iat[0])):
            if i in self.splitList:
                continue
            
            yayList = self.data[self.data["voteRecord"].str.get(i) == "+"].copy()
            nayList = self.data[self.data["voteRecord"].str.get(i) == "-"].copy()
            absList = self.data[self.data["voteRecord"].str.get(i) == "."].copy()
            
            yayRatio = yayList.size / self.data.size
            nayRatio = nayList.size / self.data.size
            absRatio = absList.size / self.data.size
            
            yayEntropy = self.entropy(yayList, '+')
            nayEntropy = self.entropy(nayList, '-')
            absEntropy = self.entropy(absList, '.')

            infoGain = currentEntropy - (yayEntropy * yayRatio + nayEntropy * nayRatio + absEntropy * absRatio)
            # print('infoGain[{0}] = {1}'.format(i, infoGain))
            if infoGain > best['val']:
                best['idx'] = i
                best['val'] = infoGain
                best['yay'] = yayList
                best['nay'] = nayList
                best['abs'] = absList
            
        return best

    def split(self):
        best = None
        currentEntropy = self.entropy(self.data, "self")
        if currentEntropy == 0 or len(self.data.index) == 0:
            self.isLeaf = self.classify
        # print("should be leaf")
            
        # print("{0} isLeaf = {1}".format(self.decision, self.isLeaf))
        # print(self.data)
        if not self.isLeaf and self.depth < self.maxDepth:
            if len(self.splitList) == len(self.data.voteRecord):
                print("splitList maxed")
                return None
            best = self.calcInfoGain(currentEntropy)
            if best['val'] == 0:
                # print("best idx", self.best['idx'])
                #if all reps have the same vote record (as the first)
                if self.data.voteRecord.eq(self.data.loc[self.data.index, 'voteRecord'].iat[0]).all():
                    count = self.data['party'].value_counts().to_frame()
                    if count['party']['D'] >= count['party']['R']:
                        self.classify = 'D'
                    else:
                        self.classify = 'R'
                    self.isLeaf = self.classify
                    return 1
                    # if self.data.party
            
            
            # print("new split on best[{0}] = {1}".format(self.best['idx'], self.best['val']))
            # self.splitIssue = chr(self.best['idx'] + 64)
            self.splitIssue = best['idx']
            if self.splitIssue == -1:
                self.isLeaf = self.classify
            
            self.splitList.append(best['idx'])
        
            self.yayNode = Node(best['yay'].copy(), self.splitList.copy(), self.depth+1, self, '+')
            self.yayNode.split()
            
            self.nayNode = Node(best['nay'].copy(), self.splitList.copy(), self.depth+1, self, '-')
            self.nayNode.split()
            
            self.absNode = Node(best['abs'].copy(), self.splitList.copy(), self.depth+1, self, '.')
            self.absNode.split()
        # else:
        #     print('leaf has best list: ')
        #     print(self.data)

        self.best = best
        
    def pruneNode(self):
    
        demCount = 0
        repCount = 0
        
        if not self.isLeaf:
        # if not self.isLeaf and self.yayNode and self.nayNode and self.absNode:
        #     if self.yayNode.classify == self.nayNode.classify == self.absNode.classify:
        #         self.isLeaf = self.yayNode.classify
            # for party in self.data["party"]:

            #     if party == "D":
            #         demCount += 1
            #     elif party == "R":
            #         repCount += 1
                    
            # if demCount > repCount:
            #     self.classify = "D"
            # elif repCount > demCount:
            #     self.classify = "R"
            # else: 
            # print('classify on prune = ', self.classify)
            self.isLeaf = self.classify
        else:
            # self.classify = None
            self.isLeaf = False
        
        return self.classify
          
    def classifyDataItem(self, item):
        res = None
        if self.isLeaf or self.splitIssue == -1:
            res = self.isLeaf
        else:
            # print("splitIssue")
            # print(self.splitIssue)
            decision = item.voteRecord[self.splitIssue]
            if decision == '+':
                res = self.yayNode.classifyDataItem(item) if self.yayNode else ''
            elif decision == '-':
                res = self.nayNode.classifyDataItem(item) if self.nayNode else ''
            elif decision == '.':
                res = self.absNode.classifyDataItem(item) if self.absNode else ''
        return res
    
    def listNodes(self, list):
        if not self.isLeaf:
            list.append(self)
            if self.yayNode is not None:
                self.yayNode.listNodes(list)
            if self.nayNode is not None:
                self.nayNode.listNodes(list)
            if self.absNode is not None:
                self.absNode.listNodes(list)
        return list
    
    def printTree(self):
        # leaf = self.isLeaf
        # while not leaf:
        
        alphaIssue = chr(self.best['idx'] + 65) if (self.splitIssue >= 0) else self.splitIssue

        print('\t'*self.depth, end='')
        print(self.decision if self.decision else '', end='')
        # print('best[{0}] val = {1}'.format(alphaIssue, self.best['val']) if alphaIssue else '', end='')
        print(' Issue {0}:'.format(alphaIssue) if (not self.isLeaf and self.splitIssue >= 0) else self.classify, end='')
        print(self.classified, self.classify)
        # print(self.data)
        if not self.isLeaf:
            # if self.best:
                # print('splitting on', self.best['idx'])
                # if(self.best['idx'] == -1):
                #     print(self.data)
            if self.yayNode:
                # print(self.best['yay'])
                self.yayNode.printTree()
            
            if self.nayNode:
                # print(self.best['nay'])
                self.nayNode.printTree()
            
            if self.absNode:
                # print(self.best['abs'])
                self.absNode.printTree()

class Tree:
    def __init__(self, data):
        self.data = data.copy()
        self.root, self.tuningSet = self.createTree(self.data)
        self.pruneTree(self.tuningSet)

    def makeTreeList(self):
        treeList = self.root.listNodes([])
        return treeList

    def pruneTree(self, tuningSet):
        tree = self.makeTreeList()
        
        if tree is None:
            print('no tree list')
            return 1
        # print('tree')
        # print(self.tree)
        # self.root.printTree()
        
        treeAccuracy = self.testTree(tuningSet)
        bestAccuracy = treeAccuracy
        bestPruneIdx = -1
        bestDepth = self.root.maxDepth
        while True:
            treeAccuracy = bestAccuracy
            # print('----------best accuracy so far ', treeAccuracy)
            for i, node in enumerate(tree):
                if node.depth == 0 or node.isLeaf:
                    continue
                node.pruneNode()
                acc = self.testTree(self.tuningSet)
                # print('acc {0} when pruning issue {1}'.format(acc, node.splitIssue))
                node.pruneNode() 
                if acc > bestAccuracy:
                    bestAccuracy = acc
                    bestPruneIdx = i
                    bestDepth = node.depth
                    # print("better accuracy at node ", i)
                elif acc == bestAccuracy:
                    if node.depth < bestDepth:
                        bestAccuracy = acc
                        bestPruneIdx = i
                        bestDepth = node.depth
                        print("same accuracy at depth", bestDepth)

            if bestPruneIdx != -1:
                tree[bestPruneIdx].pruneNode()
            
            if bestAccuracy == treeAccuracy:
                break
    
        # self.root.printTree()
        self.root.printTree()
        print('pruned accuracy = ', self.testTree(tuningSet))

    def testTree(self, testData=None):

        if testData is None:
            testData = self.tuningSet

        accurateCount = 0

        for idx, item in testData.iterrows():
            res = self.root.classifyDataItem(item)
            if res == item.party:
                accurateCount += 1

        accuracy = accurateCount / len(testData.index)

        return accuracy

    def createTree(self, data):
    
        # if data is None:
        #     data = self.data
    
        # item = data.loc[384]
        # print(data)
        
        tuningSet = data.loc[::4].copy()
        
        print(tuningSet)
        # print(tuningSet.index)
        
        newData = data.drop(tuningSet.index)
        
        print(newData)
        
        #root node split info gain
        root = Node(newData)
        best = root.split()
        # itemRes = root.classifyDataItem(item)
        # print("rep[{0}] party = {1}".format(item.rep, itemRes))
        # print("best[{0}] = {1}".format(best['idx'], best['val']))
        
        
        # self.root.printTree()
        
        
        # bacc = self.testTree(data)
        # print('accuracy on build set = ', bacc)
        # tacc = self.testTree(self.tuningSet)
        # print('accuracy on test set = ', tacc)


        
        return root, tuningSet
    
    
def crossValidate(fullData):
    
    accurateCount = 0
    
    data = fullData.copy()

    for idx, datum in data.iterrows():
        # if idx < 2:
        #     continue
        # if idx == 10:
        #     return 1
        newData = data.drop(idx)
        print(newData)
        tree = Tree(newData)
        print(tree)
        # tree.root.printTree()
        res = tree.root.classifyDataItem(datum)
        if res == datum.party:
            print('correct classify')
            accurateCount += 1
        else:
            print('wrong classify')

    accuracy = accurateCount / (len(data.index)-1)

    return accuracy
        

crossAccuracy = crossValidate(data)
print('accuracy on cross validation = ', crossAccuracy)

# newData = data.drop(2)
# tree = Tree(newData)
# # tree.root.printTree()
# print(tree.testTree())
