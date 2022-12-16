import csv
import time
import avl_skeleton
import math
import random
from re import A
from tkinter.tix import TList
import avl_skeleton
import random

"""tlist = avl_skeleton.AVLTreeList()
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
tlist2 = avl_skeleton.AVLTreeList()"""

tlist = avl_skeleton.AVLTreeList()
tlist.insert(0,"1")
print(tlist.getRoot())
tlist.insert(1,"2")
print(tlist.getRoot())
tlist.insert(2,"3")
print(tlist.getRoot())
tlist.insert(3,"4")
print(tlist.getRoot())
tlist.insert(4,"5")
print(tlist.getRoot())
