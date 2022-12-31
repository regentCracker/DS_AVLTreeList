import csv
import time
import avl_skeleton
import math
import random
from re import A
from tkinter.tix import TList
import avl_skeleton
import random
from random import randrange

"""
tlist = avl_skeleton.AVLTreeList()
print(str(tlist.insert(0,1)))
print(avl_skeleton.AVLTreeList.getMax(tlist.getRoot()).getValue())
print(tlist.insert(1,2))
print(avl_skeleton.AVLTreeList.getMax(tlist.getRoot()).getValue())
print(tlist.getRoot().right.getValue())
print(tlist.insert(2,3))
print(tlist.getRoot().left.getValue())
print(tlist.retrieve(1))
print(tlist.insert(3,4))
print(tlist.retrieve(2))
print(tlist.insert(2,5))
print(tlist.insert(3,6))
print(tlist.retrieve(4))
tlist2 = avl_skeleton.AVLTreeList()
"""

#"""
tlist = avl_skeleton.AVLTreeList()
"""
index=0
li = [17,8,22,4,12,20,26,2,6,10,14,19,21,24,28,1,3,5,7,9,11,13,15,18,23,25,27,30,0,1.5,3.5,7.5,15.5,29,31,-0.5,-1]
line=li[index]#input()
while line!=-1:
    num = float(line)
    rotations = 0
    if(tlist.length() == 0):
        rotations = tlist.insert(0, line)
        #tlist.display()
        print(str(rotations)+" rotations.")
    for i in range(tlist.length()):
        if(float(tlist.retrieve(i))>num):
            rotations = tlist.insert(i, line)
            #tlist.display()
            print(str(rotations)+" rotations.")
            break
        elif(float(tlist.retrieve(tlist.length()-1))<num):
            rotations = tlist.insert(tlist.length(), line)
            #tlist.display()
            print(str(rotations)+" rotations.")
    line=li[index]#input()
    index+=1

rotations = tlist.delete(25)
print(rotations)
tlist.display()

print(tlist.length())
print(tlist.last())
print(tlist.first())
print(tlist.listToArray())
print(tlist.getRoot())
tlist2 = tlist.permutaion()
tlist2.display()
tlist3 = tlist.sort()
tlist3.display()
print(tlist3.getRoot())

tlist4 = avl_skeleton.AVLTreeList()
for i in range(10):
    tlist4.insert(i,i)
height_diff = tlist.concat(tlist4)
tlist.display()
print(height_diff)
print(tlist.getRoot())
print(tlist.search(7))

"""


"""
for j in range(100):
    for i in range(2000):
        tlist.insert(i,str(i))
    li=[]
    counter = 0


    for i in range(2000):
        l = tlist.length
        r = randrange(l)
        rotations = tlist.delete(r)
        li.append(r)

        #print(str(rotations)+" rotations.")
        if(rotations>=7):
            counter+=1
            #tlist.display()
            #print("whoa")
            #print(li)
    print(counter)
"""




"""
tlist.insert(0,"1")
tlist.display()
tlist.insert(1,"2")
tlist.display()
tlist.insert(2,"3")
tlist.display()
tlist.insert(3,"4")
tlist.display()
tlist.insert(4,"5")
tlist.display()
tlist.insert(2,"2.5")
tlist.display()
tlist.insert(2,"2.25")
tlist.display()
tlist.insert(2,"2.125")
tlist.display()
tlist.delete(2)
tlist.display()
tlist.delete(2)
tlist.display()
"""
def chksize(lst):
    if not lst.isRealNode():
        if -1 != lst.getSize():
            print(" oi NotReal "+ str(lst.getValue()) + " " + str(lst.getSize()))
        if -1 != lst.getHeight():
            print(" oi NotReal "+ str(lst.getValue()) + " " + str(lst.getHeight()))
        return -1
    sz = chksize(lst.getRight())+chksize(lst.getLeft()) + 2
    hi = max(lst.getRight().getHeight(),lst.getLeft().getHeight()) + 1
    if sz != lst.getSize():
        print(" oi "+ str(tlist.treeRank(lst))+" "+ str(lst.getValue()) + " " + str(lst.getSize() - sz)+ " " + str(lst.getSize())+" " + str(sz))
    if hi != lst.getHeight():
        print(" vey "+ str(tlist.treeRank(lst))+" "+ str(lst.getValue()))
    if (lst.getRight().getHeight() - lst.getLeft().getHeight()) not in [-1,0,1]:
        print(" oof "+ str(tlist.treeRank(lst))+" "+ str(lst.getValue()) + " " + str(lst.getRight().getHeight() - lst.getLeft().getHeight()))
    return sz
tlist.insert(0,"0")
for i in range(20000):
    tlist.insert(randrange(tlist.length()),str(i))
print(tlist.getRoot().getSize())
tlist2 = avl_skeleton.AVLTreeList()

tlist2.insert(0,"0")
for i in range(20000):
    tlist2.insert(randrange(tlist2.length()),str(i))
tlist.concat(tlist2)
print(tlist.getRoot().getSize())
chksize(tlist.getRoot())






#"""
