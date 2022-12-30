# username - nimrodc@mail.tau.ac.il
# id1      - 200973212
# name1    - Oded Kesler
# id2      - 315427799
# name2    - Nimrod Cohen
import random

"""A class represnting a node in an AVL tree"""


class AVLNode(object):
    """Constructor, you are allowed to add more fields.

	@type value: str
	@param value: data of your node
	"""

    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None
        self.parent = None
        self.height = -1
        self.size = -1


    """returns the left child
	@rtype: AVLNode
	@returns: the left child of self, None if there is no left child
	"""

    def getLeft(self):
        return self.left


    """returns the right child

	@rtype: AVLNode
	@returns: the right child of self, None if there is no right child
	"""

    def getRight(self):
        return self.right


    """returns the parent

	@rtype: AVLNode
	@returns: the parent of self, None if there is no parent
	"""

    def getParent(self):
        return self.parent


    """return the node's value

	@rtype: str
	@returns: the value of self, None if the node is virtual
	"""

    def getValue(self):
        return self.value


    """returns the height of a node

	@rtype: int
	@returns: the height of self, -1 if the node is virtual
	"""

    def getHeight(self):
        return self.height


    """returns the size of a node

		@rtype: int
		@returns: the size of self, -1 if the node is virtual
		"""

    def getSize(self):
        return self.size


    """sets left child

	@type node: AVLNode
	@param node: a node
	"""

    def setLeft(self, node):
        self.left = node


    """sets right child

	@type node: AVLNode
	@param node: a node
	"""

    def setRight(self, node):
        self.right = node


    """sets parent

	@type node: AVLNode
	@param node: a node
	"""

    def setParent(self, node):
        self.parent = node


    """sets value

	@type value: str
	@param value: data
	"""

    def setValue(self, value):
        self.value = value


    """sets the height

	@type h: int
	@param h: the height
	"""

    def setHeight(self, h):
        self.height = h


    """sets the size of the node's subtree

		@type s: int
		@param s: the size
		"""

    def setSize(self, s):
        self.size = s


    """returns whether self is not a virtual node

	@rtype: bool
	@returns: False if self is a virtual node, True otherwise.
	"""

    def isRealNode(self):
        return self.height > -1


    """sets a node as a leaf in the tree
			"""

    def setLeaf(self):
        self.left = AVLNode("")
        self.left.setParent(self)
        self.right = AVLNode("")
        self.right.setParent(self)
        self.parent = AVLNode("")
        self.setHeight(0)
        self.setSize(0)


    """return the Balance Factor of a given node
	@rtype: int
	@returns: height differential of the children nodes
				"""

    def getBalanceFactor(self):
        return self.getLeft().getHeight() - self.getRight().getHeight()


    """sets a node as at left/right position (or new root)
				@type tree_list: AVLTreeList
				@param tree_list: The tree object where self resides
				@type new_child: AVLNode
				@param new_child: a node that will be set as a child
				@type current_child: AVLNode
				@param current_child: the current left/right child of the given node
				@pre: self.getLeft() == current_child or self.getRight() == current_child
				"""

    def setProperChild(self, tree_list, new_child, current_child):
        if not self.isRealNode():
            tree_list.setRoot(new_child)

        elif self.getRight() == current_child:
            self.setRight(new_child)

        else:
            self.setLeft(new_child)
        return


    """returns whether self is a leaf in a tree

		@rtype: bool
		@returns: False if self is not a leaf, True otherwise.
		"""

    def isLeaf(self):
        return self.getParent().isRealNode() and not self.getRight().isRealNode() and not self.getLeft().isRealNode()

    def __str__(self):
        if(self==None):
            return ""
        res = str(self.getValue())
        l = ""
        r = ""
        if(self.getLeft().isRealNode()):
            l = str(self.getLeft())
        if(self.getRight().isRealNode()):
            r = str(self.getRight())
        if(r==l and r==""):
            return res
        res += "("+l+"|"+r+")"
        return res

    def display(self):
        lines, *_ = self._display_aux()
        for line in lines:
            print(line)

    def _display_aux(self):
        """Returns list of strings, width, height, and horizontal coordinate of the root."""
        # No child.
        if self.right is None and self.left is None:
            line = '%s' % self.value
            width = len(line)
            height = 1
            middle = width // 2
            return [line], width, height, middle

        # Only left child.
        if self.right is None:
            lines, n, p, x = self.left._display_aux()
            s = '%s' % self.value
            u = len(s)
            first_line = (x + 1) * ' ' + (n - x - 1) * '_' + s
            second_line = x * ' ' + '/' + (n - x - 1 + u) * ' '
            shifted_lines = [line + u * ' ' for line in lines]
            return [first_line, second_line] + shifted_lines, n + u, p + 2, n + u // 2

        # Only right child.
        if self.left is None:
            lines, n, p, x = self.right._display_aux()
            s = '%s' % self.value
            u = len(s)
            first_line = s + x * '_' + (n - x) * ' '
            second_line = (u + x) * ' ' + '\\' + (n - x - 1) * ' '
            shifted_lines = [u * ' ' + line for line in lines]
            return [first_line, second_line] + shifted_lines, n + u, p + 2, u // 2

        # Two children.
        left, n, p, x = self.left._display_aux()
        right, m, q, y = self.right._display_aux()
        s = '%s' % self.value
        u = len(s)
        first_line = (x + 1) * ' ' + (n - x - 1) * '_' + s + y * '_' + (m - y) * ' '
        second_line = x * ' ' + '/' + (n - x - 1 + u + y) * ' ' + '\\' + (m - y - 1) * ' '
        if p < q:
            left += [n * ' '] * (q - p)
        elif q < p:
            right += [m * ' '] * (p - q)
        zipped_lines = zip(left, right)
        lines = [first_line, second_line] + [a + u * ' ' + b for a, b in zipped_lines]
        return lines, n + m + u, max(p, q) + 2, n + u // 2


"""
A class implementing the ADT list, using an AVL tree.
"""


class AVLTreeList(object):
    """
	Constructor, you are allowed to add more fields.

	"""

    def __init__(self):
        self.root = None
        self.Length = 0


    """returns whether the list is empty

	@rtype: bool
	@returns: True if the list is empty, False otherwise
	"""

    def empty(self):
        return self.root is None


    """retrieves the value of the i'th item in the list
    The function operates according to the guide lines of treeSelect in a ranked AVL tree.
    It requires walking down the height of the tree, which is O(log(n))

	@type i: int
	@pre: 0 <= i < self.length()
	@param i: index in the list
	@rtype: str
	@returns: the the value of the i'th item in the list
	"""

    def retrieve(self, i):
        if(self.Length <= i or i<0):
            return None
        return AVLTreeList.treeSelect(self.root, i).getValue()


    """inserts val at position i in the list
    When val is inserted to a non empty tree object we use either: getMaxInsert, treeSelectInsert or getPrevInsert.
    All of which require one walk down the height of the tree, which is O(log(n)).
    After the insertion, we fix and rebalance the tree, following a path from the parent of the added node up to the
    root. This route is O(log(n) long, and each required balancing takes O(1) operations.
    In total, we get a runtime complexity of O(log(n)).

	@type i: int
	@pre: 0 <= i <= self.length()
	@param i: The intended index in the list to which we insert val
	@type val: str
	@param val: the value we insert
	@rtype: int
	@returns: number of rebalancing operations
	"""

    def insert(self, i, val):
        new_node = AVLNode(val)
        new_node.setLeaf()

        if self.Length == 0:
            self.root = new_node
            self.Length += 1
            return 0

        elif i == self.Length:
            new_parent = AVLTreeList.getMaxInsert(self.root)
            new_parent.setRight(new_node)

        else:
            new_parent = AVLTreeList.treeSelectInsert(self.root, i)
            if not new_parent.getLeft().isRealNode():
                new_parent.setLeft(new_node)

            else:
                new_parent = AVLTreeList.getPrevInsert(new_parent)
                new_parent.setRight(new_node)

        new_node.setParent(new_parent)
        rotations_counter = AVLTreeList.BalanceTreeFrom(self, new_parent)
        self.Length += 1

        return rotations_counter


    """deletes the i'th item in the list by:
    1. Finding the node to be deleted by treeSelect of a ranked AVL tree, O(log(n)).
    2. Replacing the deleted node according to 4 different scenarios: it has two children, it has only a right child,
       it has only a left child, it is a leaf.
       In either cases, the longest operation is equivalent to a walk up or down the tree, and thus O(log(n)).
    3. Balancing the tree similarly (but not exactly) as we did in insert, O(log(n)).
    In total, we get a runtime complexity of O(log(n)).

	@type i: int
	@pre: 0 <= i < self.length()
	@param i: The intended index in the list to be deleted
	@rtype: int
	@returns: the number of rebalancing operation due to AVL rebalancing
	"""

    def delete(self, i):
        if(i>=self.Length):
            return -1
        if self.Length == 1:
            self.root = None
            self.Length = 0;
            return 0
        relevant_node = AVLTreeList.treeSelectDel(self.root, i)
        parent = relevant_node.getParent()
        relevant_node.setParent(AVLNode(""))

        # Deleted node has two children
        if relevant_node.getRight().isRealNode() and relevant_node.getLeft().isRealNode():

            # Get successor
            replacement = AVLTreeList.getMinDel(relevant_node.getRight())
            fix_from = replacement.getParent()
            if fix_from == relevant_node:
                fix_from = replacement;

            # Bypass successor's parent to its child
            replacement.getParent().setProperChild(self, replacement.getRight(), replacement)
            replacement.getRight().setParent(replacement.getParent())

            # Replace deleted node with successor
            replacement.setParent(parent)
            replacement.setRight(relevant_node.getRight())
            replacement.setLeft(relevant_node.getLeft())
            parent.setProperChild(self, replacement, relevant_node)

            relevant_node.setLeft(AVLNode(""))
            relevant_node.setRight(AVLNode(""))
            replacement.getRight().setParent(replacement)
            replacement.getLeft().setParent(replacement)
            replacement.setSize(replacement.getLeft().getSize() + replacement.getRight().getSize() + 2)

        else:
            fix_from = parent
            if relevant_node.isLeaf():  # Case 1 - Deleted node is a leaf
                parent.setProperChild(self, AVLNode(""), relevant_node)

            elif not relevant_node.getRight().isRealNode():  # Case 2.1 - Deleted node has no right child
                parent.setProperChild(self, relevant_node.getLeft(), relevant_node)
                relevant_node.getLeft().setParent(parent)
                relevant_node.setLeft(AVLNode(""))

            else:  # Case 2.2 - Deleted node has no left child
                parent.setProperChild(self, relevant_node.getRight(), relevant_node)
                relevant_node.getRight().setParent(parent)
                relevant_node.setRight(AVLNode(""))

        rotations_counter = AVLTreeList.BalanceTreeFrom(self, fix_from)
        self.Length -= 1

        return rotations_counter


    """returns the value of the first item in the list
    We do so by calling getMin which iteratively walks down the most left path of the tree.
    Thus, total runtime complexity of O(log(n)).

	@rtype: str
	@returns: the value of the first item, None if the list is empty
	"""

    def first(self):
        if self.length() == 0:
            return None
        return AVLTreeList.getMin(self.root).getValue()


    """returns the value of the last item in the list
    We do so by calling getMax which iteratively walks down the most right path of the tree.
    Thus, total runtime complexity of O(log(n)).

	@rtype: str
	@returns: the value of the last item, None if the list is empty
	"""

    def last(self):
        if self.length() == 0:
            return None
        return AVLTreeList.getMax(self.root).getValue()


    """returns an array representing list
    The function performs an iterative InOrder tree walk along the length of the tree.
    Thus, total runtime complexity of O(n).

	@rtype: list
	@returns: a list of strings representing the data structure
	"""

    def listToArray(self):
        result = []
        if self.Length == 0:
            return result

        stack = []
        current = self.root
        while True:
            if current.isRealNode():
                stack.append(current)
                current = current.getLeft()

            elif stack:
                current = stack.pop()
                result.append(current.getValue())
                current = current.getRight()
            else:
                break
        return result


    """returns the size of the list

	@rtype: int
	@returns: the size of the list
	"""

    def length(self):
        return self.Length

    """
    Our solution takes O(log(n)n).
    It generates a permutated list of the numbers 1-n.
    Then, for each element i in the list we insert the i-th element to the last position (log(n) to retrieve and to insert).
    @rtype: AVLTreeList
    @returns: a permutated AVLTreeList with the same elements

    """
    def permutaion(self):
        result = AVLTreeList()
        for i in AVLTreeList.perm(self.length()):
            result.insert(result.length(),self.retrieve(i-1))
        return result
    """
    Our solution takes O(log(n)n).
    First, we create a list and put in it all of the elements in the tree (this takes O(log(n)n) because we retrieve in O(log(n)) n times).
    Then, sort the list in O(log(n)n) using quick sort.
    Finally we insert n times to a new tree - every time the current index to the last position.
    @rtype: AVLTreeList
    @returns: a sorted AVLTreeList with the same elements.

    """
    def sort(self):
        result = AVLTreeList()
        nodes = [self.retrieve(i) for i in range(self.length())]
        AVLTreeList.quickSort([self.retrieve(i) for i in range(self.length())],0,len(nodes)-1)
        for node in nodes:
            result.insert(result.length(),node)
        return result




    """splits the list at the i'th index by:
    1. Finding the i'th node using treeSelect of a ranked AVL tree, O(log(n)).
    2. Splitting the tree by iterative calls of the join function from the i'th node up to the root.
    Each join takes O(log(n) in the worst case, thus the naive assumption is that the complexity is O(log(n)*O(log(n).
    However, since the tree is balanced the actual runtime complexity should be O(log(n).

	@type i: int
	@pre: 0 <= i < self.length()
	@param i: The intended index in the list according to whom we split
	@rtype: list
	@returns: a list [left, val, right], where left is an AVLTreeList representing the list until index i-1,
	right is an AVLTreeList representing the list from index i+1, and val is the value at the i'th index.
	"""

    def split(self, i):
        vars = AVLTreeList.treeSelectSplit(self.root, i)
        leftL = vars[0]
        rightL = vars[1]
        snode = vars[2]
        llst = AVLTreeList()
        llst.setRoot( leftL[len(leftL) - 1][0])
        llst.Length = llst.root.getSize() + 1
        llst.getRoot().setParent(AVLNode(""))
        print(len(leftL) - 1)
        for i in range(1,len(leftL)):
            tlst = AVLTreeList()
            tlst.setRoot(leftL[len(leftL) - i - 1][1])
            tlst.Length = tlst.getRoot().getSize() + 1
            tlst.getRoot().setParent(AVLNode(""))
            nnode = leftL[len(leftL) - i - 1][0]
            nnode.setLeaf()
            llst = AVLTreeList.join(tlst, llst, nnode)


        rlst = AVLTreeList()
        rlst.setRoot( rightL[len(rightL) - 1][0])
        rlst.Length = rlst.root.getSize() + 1
        rlst.getRoot().setParent(AVLNode(""))
        print(len(rightL) - 1)
        for i in range(1,len(rightL)):
            tlst = AVLTreeList()
            tlst.setRoot(rightL[len(rightL) - i - 1][1])
            tlst.Length = tlst.getRoot().getSize() + 1
            tlst.getRoot().setParent(AVLNode(""))
            nnode = rightL[len(rightL) - i - 1][0]
            nnode.setLeaf()
            rlst = AVLTreeList.join(rlst, tlst, nnode)

        return [llst, snode, rlst]


    """concatenates lst to self
    Given an input AVLTreeList, we append it to the end of an existing AVLTreeList.
    We do so by deleting the max node of self, and using it as the join root to the appended tree.
    This operation is done in the join function in O(log(n) runtime complexity.

	@type lst: AVLTreeList
	@param lst: a list to be concatenated after self
	@rtype: int
	@returns: the absolute value of the difference between the height of the AVL trees joined
	"""

    def concat(self, lst):
        original_height_diff = self.root.getHeight() - lst.root.getHeight()
        new_root = AVLTreeList.getMax(self.root)
        self.delete(self.Length - 1)
        self.join(lst, new_root)
        return original_height_diff


    """concatenates self to lst2, using the input node between them.
    In other words, it does the heavy lifting that is needed for the split and concat operations.
    The function operates according to 3 different cases:
    1. Same height for both trees, in which minimal adjustments are needed.
    2. Self is higher than appended AVLTreeList.
    3. Self is shorter than appended AVLTreeList

    For cases 2 and 3 we need to walk down the relevant tree to find the relevant node for the join, O(log(n)).
    Other than several adjustments, after the join we perform rebalancing operations up the tree, O(log(n)).
    Thus, in total we get runtime complexity of O(log(n)).

    @type lst2: AVLTreeList
	@param lst2: a AVLTreeList to be concatenated after self
	@type new_root: AVLNode
	@param new_root: the root of the join (not necessarily the root of the returning tree)
	@rtype: AVLTreeList
	@returns: self
	"""

    def join(self, lst2, new_root):
        height_diff = self.root.getHeight() - lst2.root.getHeight()
        new_root.setLeaf()
        if lst2.Length == 0:
            if self.Length == 0:
                self.setRoot(new_root)
                self.Length = 1
                return self
            parent = self.getMaxInsert(self.getRoot())
            parent.setRight(new_root)
            new_root.setParent(parent)
            new_root.setSize(new_root.getRight().getSize() + new_root.getLeft().getSize() + 2)
            self.Length += 1
            AVLTreeList.BalanceTreeFrom(self,parent)
            return self
        elif self.Length == 0:
            parent = AVLTreeList.treeSelectInsert(lst2.getRoot(), 0)
            parent.setLeft(new_root)
            new_root.setParent(parent)
            lst2.Length += 1
            new_root.setSize(new_root.getRight().getSize() + new_root.getLeft().getSize() + 2)
            AVLTreeList.BalanceTreeFrom(lst2, parent)
            self = lst2
            return self


        if height_diff == 0:
            new_root.setLeft(self.root)
            self.root.setParent(new_root)
            new_root.setRight(lst2.root)
            new_root.setSize(new_root.getRight().getSize() + new_root.getLeft().getSize() + 2)

            lst2.root.setParent(new_root)
            new_root.setHeight(max(new_root.getLeft().getHeight(), new_root.getRight().getHeight()) + 1)
            self.setRoot(new_root)

        elif height_diff < 0:
            left_height = self.root.getHeight()
            start = lst2.root

            while start.getHeight() > left_height:
                start.setSize(start.getSize() + self.Length + 1)
                start = start.getLeft()

            new_root.setParent(start.getParent())
            new_root.getParent().setLeft(new_root)
            new_root.setRight(start)
            start.setParent(new_root)
            new_root.setLeft(self.root)
            self.root.setParent(new_root)
            new_root.setSize(new_root.getLeft().getSize()+new_root.getRight().getSize()+2)
            new_root.setHeight(self.root.getHeight() + 1)

            #AVLTreeList.fixHeight(new_root)
            AVLTreeList.BalanceTreeFrom(self, new_root.getParent())

            self.setRoot(lst2.root)
            lst2.setRoot(None)

        else:
            right_height = lst2.root.getHeight()
            start = self.root

            while start.getHeight() > right_height:
                start.setSize(start.getSize() + lst2.Length + 1)
                start = start.getRight()

            new_root.setParent(start.getParent())
            new_root.getParent().setRight(new_root)
            new_root.setLeft(start)
            start.setParent(new_root)
            new_root.setRight(lst2.root)
            lst2.root.setParent(new_root)
            new_root.setSize(new_root.getLeft().getSize()+new_root.getRight().getSize()+2)
            new_root.setHeight(lst2.root.getHeight() + 1)

            AVLTreeList.BalanceTreeFrom(self, new_root.getParent())

            lst2.setRoot(None)

        self.Length = lst2.Length + self.Length + 1
        return self


    """searches for a *value* in the list
    By calling inOrderSearch that performs an InOrder tree walk, until it gets to the relevant node.
    Thus, worst runtime complexity of O(log(n).

	@type val: str
	@param val: a value to be searched
	@rtype: int
	@returns: the first index that contains val, -1 if not found.
	"""

    def search(self, val):
        return AVLTreeList.inOrderSearch(self.root, val) or -1


    """returns the root of the tree representing the list

	@rtype: AVLNode
	@returns: the root, None if the list is empty
	"""

    def getRoot(self):
        return self.root


    """set a new root

		@type val: AVLNode
		@param val: The new root to the tree list
		"""

    def setRoot(self, node):
        self.root = node


    """returns the max node of a subtree, starting from it's root
    Walks down the rightest path of a tree, O(log(n).

        @type some_root: AVLNode
		@param some_root: The node to start walking from
		@rtype: AVLNode
		@returns: the max node in a tree
		"""

    @staticmethod
    def getMax(some_root):
        while some_root.getRight().isRealNode():
            some_root = some_root.getRight()
        return some_root


    """returns the max node of a subtree, starting from it's root, and updating the size for each node it passes
    Walks down the rightest path of a tree, O(log(n).

        @type some_root: AVLNode
		@param some_root: The node to start walking from
		@rtype: AVLNode
		@returns: the max node in a tree
		"""

    @staticmethod
    def getMaxInsert(some_root):
        some_root.setSize(some_root.getSize()+1)
        while some_root.getRight().isRealNode():
            some_root = some_root.getRight()
            some_root.setSize(some_root.getSize()+1)
        return some_root


    """returns the min node of a subtree, starting from it's root
    Walks down the leftest path of a tree, O(log(n).

            @type some_root: AVLNode
		    @param some_root: The node to start walking from
			@rtype: AVLNode
			@returns: the min node in a tree
			"""

    @staticmethod
    def getMin(some_root):
        while some_root.getLeft().isRealNode():
            some_root = some_root.getLeft()
        return some_root


    """returns the min node of a subtree, starting from it's root and deducts 1 from the height of all nodes passed.
    Walks down the leftest path of a tree, O(log(n).

            @type some_root: AVLNode
		    @param some_root: The node to start walking from
			@rtype: AVLNode
			@returns: the min node in a tree
			"""

    @staticmethod
    def getMinDel(some_root):
        some_root.setSize(some_root.getSize() - 1)
        while some_root.getLeft().isRealNode():
            some_root = some_root.getLeft()
            some_root.setSize(some_root.getSize() - 1)
        return some_root


    """returns the node at i'th index of the list
    By applying TreeSelect of a ranked AVL tree, walking down the height of the tree, O(log(n).

			@pre: 0 <= i <= self.length()
			@type some_root: AVLNode
			@param some_root: the root to start the search from
			@type i: int
			@param i: rank of desired node
			@rtype: AVLNode
			@returns: the node at the given rank
			"""

    @staticmethod
    def treeSelect(some_root, i):
        counter = some_root.getLeft().getSize() + 1
        if i == counter:
            return some_root

        elif i < counter:
            return AVLTreeList.treeSelect(some_root.getLeft(), i)
        else:
            return AVLTreeList.treeSelect(some_root.getRight(), i - counter - 1)


    """returns the node at i'th index of the list and adjusts the sizes of the parent nodes
    By applying TreeSelect of a ranked AVL tree, walking down the height of the tree, O(log(n).

			@pre: 0 <= rank <= self.length()
			@type some_root: AVLNode
			@param some_root: the root to start the search from
			@type rank: int
			@param int: rank of desired node
			@rtype: AVLNode
			@returns: the node at the given rank
			"""

    @staticmethod
    def treeSelectInsert(some_root, rank):
        counter = some_root.getLeft().getSize() + 1
        some_root.setSize(some_root.getSize() + 1)

        if rank == counter:
            return some_root
        elif rank < counter:
            return AVLTreeList.treeSelectInsert(some_root.getLeft(), rank)
        else:
            return AVLTreeList.treeSelectInsert(some_root.getRight(), rank - counter - 1)


    """returns a list containing the required methods for a split operation
    Operations are done by O(1) walking down the height of the tree, O(log(n).
			@pre: 1 <= rank <= self.length()
			@type some_root: AVLNode
			@param some_root: the root to start the search from
			@type rank: int
			@param int: rank of desired node
			@rtype: List
			@returns: a list [left, right, x] where x is the node at i'th index of the list (which is the node at rank i +1) , and right and left are lists of arrays
            of size 2 containing the nodes and corresponding trees in the right and left sides of a split of self at x respectively.
			"""

    @staticmethod
    def treeSelectSplit(some_root, rank):

        counter = some_root.getLeft().getSize() + 1
        x = some_root
        left = []
        right = []
        while True:
            if rank == counter:
                x = some_root
                right.append([x.getRight()])
                left.append([x.getLeft()])
                break
            elif rank < counter:
                right.append([some_root, some_root.getRight()])
                some_root = some_root.getLeft()
                counter = some_root.getLeft().getSize() + 1
            else:
                left.append([some_root, some_root.getLeft()])
                rank = rank - counter - 1
                some_root = some_root.getRight()
                counter = some_root.getLeft().getSize() + 1

        return [left, right, x]


    """returns the node at i'th index of the list and adjusts the sizes of the parent nodes
        By applying TreeSelect of a ranked AVL tree, walking down the height of the tree, O(log(n).

    			@pre: 0 <= rank <= self.length()
    			@type some_root: AVLNode
    			@param some_root: the root to start the search from
    			@type rank: int
    			@param int: rank of desired node
    			@rtype: AVLNode
    			@returns: the node at the given rank
    			"""

    @staticmethod
    def treeSelectDel(some_root, rank):
        counter = some_root.getLeft().getSize() + 1
        some_root.setSize(some_root.getSize() - 1)

        if rank == counter:
            return some_root
        elif rank < counter:
            return AVLTreeList.treeSelectDel(some_root.getLeft(), rank)
        else:
            return AVLTreeList.treeSelectDel(some_root.getRight(), rank - counter - 1)


    """returns the predecessor of a node in the tree and fixes height and sizes
    Deals with two cases, in which both are done in a single walk up or down the tree, O(log(n).

			@type node: AVLNode
			@param node: node at rank i
			@rtype: AVLNode
			@returns: node at rank i - 1
			"""

    @staticmethod
    def getPrevInsert(node):
        if node.getLeft().isRealNode():
            return AVLTreeList.getMaxInsert(node.getLeft())

        while node == node.getParent().getLeft():
            node.setSize(node.getSize()-1)
            node = node.getParent()

        if not node.getParent().isRealNode():
            return None  # i.e. this node is the minimal node of the tree
        return node.getParent()


    """returns the first index of a given value
    Perform a recursive implementation of an InOrder tree walk until it finds the reelvant node (if present in tree)
    Worst case runtime complexity of O(n).

				@type node: AVLNode
				@param node: the node to start the walk from
				@type value: string
				@param value: the value to search in the tree nodes
				@rtype: int
				@returns: list index of first appearance of a given value, -1 if value not found
				"""

    @staticmethod
    def inOrderSearch(node, value):
        if node.isRealNode():

            v = AVLTreeList.inOrderSearch(node.getLeft(), value)
            if v >-1:
                return v
            if node.getValue() == value:
                return  node.getLeft().getSize() + 1
            v = AVLTreeList.inOrderSearch(node.getRight(), value)
            if v >-1:
                return v + node.getLeft().getSize() + 2

        return -1

    """returns the index of a given value
    Finding the rank of a node in an ranked AVL tree,by walking down the height of the tree, O(log(n)).

				@type node: AVLNode
				@param tree: the node to start the walk from
				@rtype: int
				@returns: the rank of a node in its tree
				"""

    @staticmethod
    def treeRank(node):
        counter = node.getLeft().getSize() + 1

        while node.getParent().isRealNode():
            parent = node.getParent()
            if node == parent.getRight():
                counter += parent.getLeft().getSize() + 1
            node = parent

        return counter


    """Balance an AVL tree, after performing structural changes in it.
    The function prepares the data for the proper rotations that need to be done from the first structural change point,
    up to the root. The route is log(n) long, and each rotation takes O(1).
    Thus, in total it operates in worst case runtime complexity of O(log(n)).

						@type tree_list: AVLTreeList
						@param tree_list: a pointer to the tree list
						@type node: AVLNode
						@pre: node.isRealNode() = True
						@param node: the node to start the walk from
						@rtype: int
						@returns: total number of rotations done to balance the tree
						"""

    @staticmethod
    def BalanceTreeFrom(tree_list, node):
        counter = 0
        while node.isRealNode():
            #prev = node.getHeight()
            node.setHeight(max(node.getLeft().getHeight(),node.getRight().getHeight()) + 1)
            #if prev != node.getHeight():
                #counter += 1
            balance_factor = node.getBalanceFactor()

            if balance_factor == -2:
                counter += AVLTreeList.rotate(tree_list, node, -2, node.getRight(), node.getRight().getBalanceFactor())

            if balance_factor == 2:
                counter += AVLTreeList.rotate(tree_list, node, 2, node.getLeft(), node.getLeft().getBalanceFactor())

            node = node.getParent()
        return counter


    """Perform rotations to balance an AVL tree according to a node, its relevant child, and their balance factors.
    The functions that use this method, input the relevant node and child according to the desired action:
    insertion or deletion, and the balance factors differentiation.
    When calling this method, we apply 1 of 4 possible rotations, and each rotation uses getters and setters functions.
    Thus, as a whole, it runs in complexity of O(1).

						@type tree_list: AVLTreeList
						@param tree_list: a pointer to the tree list
						@type node: AVLNode
						@param node: the node to start rotating from
						@type bf: int
						@param bf: balance factor of the given node
						@type child: AVLNode
						@param child: relevant child node
						@type child_bf: int
						@param child_bf: relevant child's balance factor
						@rtype: int
						@returns: number of rotations done
							"""

    @staticmethod
    def rotate(tree_list, node, bf, child, child_bf):

        if bf == 2 and child_bf in [1, 0]:  # LL rotation
            node.setLeft(child.getRight())

            node.getLeft().setParent(node)
            child.setRight(node)
            child.setParent(node.getParent())
            child.getParent().setProperChild(tree_list, child, node)
            node.setParent(child)
            node.setSize(node.getRight().getSize() + node.getLeft().getSize() + 2)
            child.setSize(child.getRight().getSize() + child.getLeft().getSize() + 2)
            node.setHeight(max(node.getLeft().getHeight(), node.getRight().getHeight()) + 1)
            return 1

        if bf == -2 and child_bf in [-1, 0]:  # Left rotation
            node.setRight(child.getLeft())
            node.getRight().setParent(node)
            child.setLeft(node)
            child.setParent(node.getParent())
            child.getParent().setProperChild(tree_list, child, node)
            node.setParent(child)
            node.setSize(node.getRight().getSize() + node.getLeft().getSize() + 2)
            child.setSize(child.getRight().getSize() + child.getLeft().getSize() + 2)
            node.setHeight(max(node.getLeft().getHeight(), node.getRight().getHeight()) + 1)
            return 1

        if bf == 2 and child_bf == -1:  # Left then right rotation
            node.setLeft(child.getRight().getRight())
            child.getRight().getRight().setParent(node)
            child.getRight().setRight(node)
            child.getRight().getLeft().setParent(child)
            child.getRight().setParent(node.getParent())
            node.getParent().setProperChild(tree_list, child.getRight(), node)
            node.setParent(child.getRight())
            child.setRight(node.getParent().getLeft())
            child.setParent(node.getParent())
            node.getParent().setLeft(child)
            node.setSize(node.getRight().getSize() + node.getLeft().getSize() + 2)
            child.setSize(child.getRight().getSize() + child.getLeft().getSize() + 2)
            node.getParent().setSize(node.getParent().getRight().getSize() + node.getParent().getLeft().getSize() + 2)
            child.setHeight(max(child.getLeft().getHeight(), child.getRight().getHeight()) + 1)
            node.setHeight(max(node.getLeft().getHeight(), node.getRight().getHeight()) + 1)
            node.getParent().setHeight(max(node.getParent().getLeft().getHeight(), node.getParent().getRight().getHeight()) + 1)
            return 2

        if bf == -2 and child_bf == 1:  # Right then left rotation
            node.setRight(child.getLeft().getLeft())
            child.getLeft().getLeft().setParent(node)
            child.getLeft().setLeft(node)
            child.getLeft().getRight().setParent(child)
            child.getLeft().setParent(node.getParent())
            node.getParent().setProperChild(tree_list, child.getLeft(), node)
            node.setParent(child.getLeft())
            child.setLeft(node.getParent().getRight())
            child.setParent(node.getParent())
            node.getParent().setRight(child)
            node.setSize(node.getRight().getSize() + node.getLeft().getSize() + 2)
            child.setSize(child.getRight().getSize() + child.getLeft().getSize() + 2)
            node.getParent().setSize(node.getParent().getRight().getSize() + node.getParent().getLeft().getSize() + 2)
            child.setHeight(max(child.getLeft().getHeight(), child.getRight().getHeight()) + 1)
            node.setHeight(max(node.getLeft().getHeight(), node.getRight().getHeight()) + 1)
            node.getParent().setHeight(max(node.getParent().getLeft().getHeight(), node.getParent().getRight().getHeight()) + 1)
            return 2
        if(node is tree_list.getRoot()):
            tree_list.setRoot(node.getParent())
    def display(self):
        if(self.Length == 0):
            print("<EmptyList>")
            return
        self.getRoot().display()
    """
	@type n: int
    @param tree_list: length of returned list
    @pre: n>0
    @rtype: list
    @returns: a permutation of the list [1,...,n]
    """
    @staticmethod
    def perm(n):
        res = [i for i in range(1,n+1)]
        for pivot in range(1,n):
            switch = (random.randint(0,pivot),pivot)
            res[switch[0]], res[switch[1]] = res[switch[1]], res[switch[0]]
        return res
    """
    function to find the partition for the quicksort

    @type array: list of elements
    @array: the list to be sorted
    @pre: array is totally an ordered set.
    @type low: int
    @param low: index to start partition from
    @type high: int
    @param high: index to stop partition from
    @rtype: int
    @returns: the partition position
    """
    @staticmethod
    def partition(array, low, high):
      pivot = array[high]
      i = low - 1
      for j in range(low, high):
        if array[j] <= pivot:
          i = i + 1
          (array[i], array[j]) = (array[j], array[i])
      (array[i + 1], array[high]) = (array[high], array[i + 1])
      return i + 1

    """
    function to perform quicksort

    @type array: list of elements
    @array: the list to be sorted
    @pre: array is totally an ordered set.
    @type low: int
    @param low: index to start sorting from
    @type high: int
    @param high: index to stop sorting from
    @rtype: None
    @post: array should be sorted from lowest to highest
    """
    @staticmethod
    def quickSort(array, low, high):
      if low < high:
        pi = AVLTreeList.partition(array, low, high)
        AVLTreeList.quickSort(array, low, pi - 1)
        AVLTreeList.quickSort(array, pi + 1, high)
