《剑指 Offer》


#### [面试题03\. 数组中重复的数字](https://leetcode-cn.com/problems/shu-zu-zhong-zhong-fu-de-shu-zi-lcof/)
找出数组中重复的数字。

在一个长度为 n 的数组 nums 里的所有数字都在 0～n-1 的范围内。数组中某些数字是重复的，但不知道有几个数字重复了，也不知道每个数字重复了几次。请找出数组中任意一个重复的数字。

示例 1：
输入：[2, 3, 1, 0, 2, 5, 3]
输出：2 或 3

限制：
2 <= n <= 100000

```
class Solution:
    def findRepeatNumber(self, nums: List[int]) -> int:
        for i in range(len(nums)):
            while nums[i] != i:
                if nums[nums[i]] == nums[i]:
                    return nums[i]
                nums[nums[i]], nums[i] = nums[i], nums[nums[i]]
        return -1

        # i = 0 
        # while i < len(nums):
        #     if nums[i] == i: 
        #         i += 1
        #         continue
        #     if nums[nums[i]] == nums[i]: return nums[i]
        #     nums[nums[i]], nums[i] = nums[i], nums[nums[i]]
        # return -1
```
---

#### [面试题04\. 二维数组中的查找](https://leetcode-cn.com/problems/er-wei-shu-zu-zhong-de-cha-zhao-lcof/)

在一个 n * m 的二维数组中，每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序。请完成一个函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。

### 示例:

现有矩阵 matrix 如下：

[
  [1,   4,  7, 11, 15],
  [2,   5,  8, 12, 19],
  [3,   6,  9, 16, 22],
  [10, 13, 14, 17, 24],
  [18, 21, 23, 26, 30]
]
给定 target = 5，返回 true。

给定 target = 20，返回 false。

### 限制：

0 <= n <= 1000

0 <= m <= 1000

```
class Solution:
    def findNumberIn2DArray(self, matrix: List[List[int]], target: int) -> bool:
        i, j = len(matrix)-1, 0
        while i >= 0 and j < len(matrix[0]):
            if matrix[i][j] == target: return True
            elif matrix[i][j] > target: i -= 1
            else: j += 1
        return False
```
注意：本题与主站 240 题相同：https://leetcode-cn.com/problems/search-a-2d-matrix-ii/

---

#### [面试题05\. 替换空格](https://leetcode-cn.com/problems/ti-huan-kong-ge-lcof/)

请实现一个函数，把字符串 s 中的每个空格替换成"%20"。

### 示例 1：

输入：s = "We are happy."
输出："We%20are%20happy."
 
###限制：

0 <= s 的长度 <= 10000

```
class Solution:
    def replaceSpace(self, s: str) -> str:
        res = []
        for c in s:
            if c == ' ': res.append("%20")
            else: res.append(c)
        return "".join(res)
```
---
#### [面试题06\. 从尾到头打印链表](https://leetcode-cn.com/problems/cong-wei-dao-tou-da-yin-lian-biao-lcof/)

输入一个链表的头节点，从尾到头反过来返回每个节点的值（用数组返回）。

### 示例 1：

输入：head = [1,3,2]
输出：[2,3,1]

### 限制：
0 <= 链表长度 <= 10000

```
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def reversePrint(self, head: ListNode) -> List[int]:
        # 递归
        # return self.reversePrint(head.next) + [head.val] if head else []

        # 辅助栈法
        stack = []
        while head:
            stack.append(head.val)
            head = head.next
        return stack[::-1]
```
---
#### [面试题07\. 重建二叉树](https://leetcode-cn.com/problems/zhong-jian-er-cha-shu-lcof/)
输入某二叉树的前序遍历和中序遍历的结果，请重建该二叉树。假设输入的前序遍历和中序遍历的结果中都不含重复的数字。

例如，给出

前序遍历 preorder = [3,9,20,15,7]
中序遍历 inorder = [9,3,15,20,7]
返回如下的二叉树：

```
    3
   / \
  9  20
     /  \
   15   7
 ```
限制：
0 <= 节点个数 <= 5000

注意：本题与主站 105 题重复：https://leetcode-cn.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/

---

```
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
        self.dic, self.pre = {}, preorder
        for i in range(len(inorder)):
            self.dic[inorder[i]] = i
        return self.helper(0, 0, len(inorder)-1)
    def helper(self, pre_root, in_left, in_right):
        if in_left > in_right: return
        root = TreeNode(self.pre[pre_root])
        i = self.dic[self.pre[pre_root]]
        root.left = self.helper(pre_root+1, in_left, i-1)
        root.right = self.helper(i-in_left+pre_root+1, i+1, in_right)
        return root

        # if inorder:
        #     index = inorder.index(preorder.pop(0))
        #     root = TreeNode(inorder[index])
        #     root.left = self.buildTree(preorder, inorder[0:index])
        #     root.right = self.buildTree(preorder, inorder[index+1:])
        #     return root

        # if not preorder or not inorder:
        #     return None
        # if len(preorder) != len(inorder):
        #     return None
        # root = TreeNode(preorder[0])
        # index = inorder.index(preorder[0])
        # root.left = self.buildTree(preorder[1:index+1], inorder[:index])
        # root.right = self.buildTree(preorder[index+1:], inorder[index+1:]) 
        # return root
```
---
#### [面试题09\. 用两个栈实现队列](https://leetcode-cn.com/problems/yong-liang-ge-zhan-shi-xian-dui-lie-lcof/)
用两个栈实现一个队列。队列的声明如下，请实现它的两个函数 appendTail 和 deleteHead ，分别完成在队列尾部插入整数和在队列头部删除整数的功能。(若队列中没有元素，deleteHead 操作返回 -1 )

 

示例 1：

输入：
["CQueue","appendTail","deleteHead","deleteHead"]
[[],[3],[],[]]
输出：[null,null,3,-1]
示例 2：

输入：
["CQueue","deleteHead","appendTail","appendTail","deleteHead","deleteHead"]
[[],[],[5],[2],[],[]]
输出：[null,-1,null,null,5,2]
提示：

1 <= values <= 10000
最多会对 appendTail、deleteHead 进行 10000 次调用

```
class CQueue:

    def __init__(self):
        self.A , self.B = [], []
        
    def appendTail(self, value: int) -> None:
        self.A.append(value)

    def deleteHead(self) -> int:
        if self.B: return self.B.pop()
        if not self.A: return -1
        while self.A:
            self.B.append(self.A.pop())
        return self.B.pop()
        
# Your CQueue object will be instantiated and called as such:
# obj = CQueue()
# obj.appendTail(value)
# param_2 = obj.deleteHead()
```
---
#### [面试题10- I. 斐波那契数列](https://leetcode-cn.com/problems/fei-bo-na-qi-shu-lie-lcof/)
写一个函数，输入 n ，求斐波那契（Fibonacci）数列的第 n 项。斐波那契数列的定义如下：

F(0) = 0,   F(1) = 1
F(N) = F(N - 1) + F(N - 2), 其中 N > 1.
斐波那契数列由 0 和 1 开始，之后的斐波那契数就是由之前的两数相加而得出。

答案需要取模 1e9+7（1000000007），如计算初始结果为：1000000008，请返回 1。

示例 1：

输入：n = 2
输出：1
示例 2：

输入：n = 5
输出：5
 
提示：
0 <= n <= 100
注意：本题与主站 509 题相同：https://leetcode-cn.com/problems/fibonacci-number/

```
class Solution:
    def fib(self, n: int) -> int:
        a, b = 0, 1
        for x in range(n):
            a, b = b, a+b
        return a % 1000000007

# import functools
# class Solution:
#     @functools.lru_cache
#     def fib(self, n: int) -> int:
#         return n if n<2 else (self.fib(n-1) + self.fib(n-2)) % 1000000007
```
---
#### [面试题10- II. 青蛙跳台阶问题](https://leetcode-cn.com/problems/qing-wa-tiao-tai-jie-wen-ti-lcof/)
一只青蛙一次可以跳上1级台阶，也可以跳上2级台阶。求该青蛙跳上一个 n 级的台阶总共有多少种跳法。

答案需要取模 1e9+7（1000000007），如计算初始结果为：1000000008，请返回 1。

示例 1：

输入：n = 2
输出：2
示例 2：

输入：n = 7
输出：21
提示：

0 <= n <= 100
注意：本题与主站 509 题相同：https://leetcode-cn.com/problems/fibonacci-number/

```
class Solution:
    def numWays(self, n: int) -> int:
        a, b = 1, 1
        for _ in range(n):
            a, b = b, a+b
        return a % 1000000007
        
# import functools
# class Solution:
#     @functools.lru_cache
#     def numWays(self, n: int) -> int:
#         return 1 if n < 2 else (self.numWays(n-1)+self.numWays(n-2)) % 1000000007
```
---
#### [面试题11\. 旋转数组的最小数字](https://leetcode-cn.com/problems/xuan-zhuan-shu-zu-de-zui-xiao-shu-zi-lcof/)
把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。输入一个递增排序的数组的一个旋转，输出旋转数组的最小元素。例如，数组 [3,4,5,1,2] 为 [1,2,3,4,5] 的一个旋转，该数组的最小值为1。  

示例 1：

输入：[3,4,5,1,2]
输出：1
示例 2：

输入：[2,2,2,0,1]
输出：0
注意：本题与主站 154 题相同：https://leetcode-cn.com/problems/find-minimum-in-rotated-sorted-array-ii/


```
class Solution:
    def minArray(self, numbers: List[int]) -> int:
        left, right = 0, len(numbers)-1
        while left < right:
            mid = (left + right) >> 1
            if numbers[mid] > numbers[right]: left = mid + 1
            elif numbers[mid] < numbers[right]: right = mid
            else: right -= 1 
        return numbers[left]
```
---        
#### [面试题12\. 矩阵中的路径](https://leetcode-cn.com/problems/ju-zhen-zhong-de-lu-jing-lcof/)
请设计一个函数，用来判断在一个矩阵中是否存在一条包含某字符串所有字符的路径。路径可以从矩阵中的任意一格开始，每一步可以在矩阵中向左、右、上、下移动一格。如果一条路径经过了矩阵的某一格，那么该路径不能再次进入该格子。例如，在下面的3×4的矩阵中包含一条字符串“bfce”的路径（路径中的字母用加粗标出）。

[["a","b","c","e"],
["s","f","c","s"],
["a","d","e","e"]]

但矩阵中不包含字符串“abfb”的路径，因为字符串的第一个字符b占据了矩阵中的第一行第二个格子之后，路径不能再次进入这个格子。

 

示例 1：

输入：board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCCED"
输出：true
示例 2：

输入：board = [["a","b"],["c","d"]], word = "abcd"
输出：false
提示：

1 <= board.length <= 200
1 <= board[i].length <= 200
注意：本题与主站 79 题相同：https://leetcode-cn.com/problems/word-search/


```
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        def dfs(i, j, k):
            if not 0 <= i <len(board) or not 0 <= j < len(board[0]) or board[i][j] != word[k] : return False
            if k == len(word) - 1 : return True
            tmp, board[i][j] = board[i][j], '/'
            res = dfs(i+1, j, k+1) or dfs(i-1, j, k+1) or dfs(i, j+1, k+1) or dfs(i, j-1, k+1)
            board[i][j] = tmp
            return res

        for i in range(len(board)):
            for j in range(len(board[0])):
                if dfs(i, j, 0): return True
        return False
```
---
#### [面试题13\. 机器人的运动范围](https://leetcode-cn.com/problems/ji-qi-ren-de-yun-dong-fan-wei-lcof/)
地上有一个m行n列的方格，从坐标 [0,0] 到坐标 [m-1,n-1] 。一个机器人从坐标 [0, 0] 的格子开始移动，它每次可以向左、右、上、下移动一格（不能移动到方格外），也不能进入行坐标和列坐标的数位之和大于k的格子。例如，当k为18时，机器人能够进入方格 [35, 37] ，因为3+5+3+7=18。但它不能进入方格 [35, 38]，因为3+5+3+8=19。请问该机器人能够到达多少个格子？

 

示例 1：

输入：m = 2, n = 3, k = 1
输出：3
示例 1：

输入：m = 3, n = 1, k = 0
输出：1
提示：

1 <= n,m <= 100
0 <= k <= 20


```
class Solution:
    def movingCount(self, m: int, n: int, k: int) -> int:
        """dfs"""
        def sum(i, j):
            tmp = 0
            while i:
                tmp += i % 10
                i //= 10
            while j:
                tmp += j % 10
                j //= 10
            return tmp
        def dfs(i, j):
            nonlocal res
            if (i, j) in marked or i == m or j == n or sum(i, j) > k: return 
            marked.add((i, j))
            res += 1
            dfs(i+1, j), dfs(i, j+1)
        marked = set()
        res = 0
        dfs(0, 0)
        return res

# class Solution:
#     def sum(self, row, col):
#         tmp = 0
#         while row:
#             tmp += row % 10
#             row //= 10
#         while col:
#             tmp += col % 10
#             col //= 10
#         return tmp
#     def movingCount(self, m: int, n: int, k: int) -> int:
#         """BFS"""
#         marked = set()
#         deque = collections.deque()
#         deque.append((0,0))
#         while deque:
#             x, y = deque.popleft()
#             if (x, y) not in marked and self.sum(x, y) <= k:
#                 marked.add((x,y))
#                 for dx, dy in [(1, 0), (0, 1)]:
#                     if 0 <= x+dx < m and 0 <= y+dy < n:
#                         deque.append((x+dx, y+dy))
#         return len(marked)

```
---
#### [面试题14- I. 剪绳子](https://leetcode-cn.com/problems/jian-sheng-zi-lcof/)
给你一根长度为 n 的绳子，请把绳子剪成整数长度的 m 段（m、n都是整数，n>1并且m>1），每段绳子的长度记为 k[0],k[1]...k[m] 。请问 k[0]*k[1]*...*k[m] 可能的最大乘积是多少？例如，当绳子的长度是8时，我们把它剪成长度分别为2、3、3的三段，此时得到的最大乘积是18。

示例 1：

输入: 2
输出: 1
解释: 2 = 1 + 1, 1 × 1 = 1
示例 2:

输入: 10
输出: 36
解释: 10 = 3 + 3 + 4, 3 × 3 × 4 = 36
提示：

2 <= n <= 58
注意：本题与主站 343 题相同：https://leetcode-cn.com/problems/integer-break/


```
class Solution:
    def cuttingRope(self, n: int) -> int:
        if n <=3: return n-1
        a, b = n // 3, n % 3
        if b == 0:   return 3 ** a
        elif b == 1: return 3 ** (a-1) * 4
        elif b == 2: return 3 ** a * 2
```
---
#### [面试题14- II. 剪绳子 II](https://leetcode-cn.com/problems/jian-sheng-zi-ii-lcof/)
给你一根长度为 n 的绳子，请把绳子剪成整数长度的 m 段（m、n都是整数，n>1并且m>1），每段绳子的长度记为 k[0],k[1]...k[m] 。请问 k[0]*k[1]*...*k[m] 可能的最大乘积是多少？例如，当绳子的长度是8时，我们把它剪成长度分别为2、3、3的三段，此时得到的最大乘积是18。

答案需要取模 1e9+7（1000000007），如计算初始结果为：1000000008，请返回 1。

 

示例 1：

输入: 2
输出: 1
解释: 2 = 1 + 1, 1 × 1 = 1
示例 2:

输入: 10
输出: 36
解释: 10 = 3 + 3 + 4, 3 × 3 × 4 = 36
 

提示：

2 <= n <= 1000
注意：本题与主站 343 题相同：https://leetcode-cn.com/problems/integer-break/

```
class Solution:
    def cuttingRope(self, n: int) -> int:
        if n <= 3: return n-1
        a, b, p = n // 3, n % 3, 1000000007
        if b == 0: return 3 ** a % p
        elif b == 1: return 3 ** (a-1) * 4 % p
        elif b == 2: return 3 ** a * 2 % p
```
---
#### [面试题15\. 二进制中1的个数](https://leetcode-cn.com/problems/er-jin-zhi-zhong-1de-ge-shu-lcof/)
请实现一个函数，输入一个整数，输出该数二进制表示中 1 的个数。例如，把 9 表示成二进制是 1001，有 2 位是 1。因此，如果输入 9，则该函数输出 2。

示例 1：

输入：00000000000000000000000000001011
输出：3
解释：输入的二进制串 00000000000000000000000000001011 中，共有三位为 '1'。
示例 2：

输入：00000000000000000000000010000000
输出：1
解释：输入的二进制串 00000000000000000000000010000000 中，共有一位为 '1'。
示例 3：

输入：11111111111111111111111111111101
输出：31
解释：输入的二进制串 11111111111111111111111111111101 中，共有 31 位为 '1'。
 

注意：本题与主站 191 题相同：https://leetcode-cn.com/problems/number-of-1-bits/


```
class Solution:
    def hammingWeight(self, n: int) -> int:
        res = 0
        while n:
            res += n & 1 
            n >>= 1
        return res

        # return bin(n).count('1')

        # count = 0
        # while n:
        #     n &= n-1
        #     count += 1
        # return count
```
---
#### [面试题16\. 数值的整数次方](https://leetcode-cn.com/problems/shu-zhi-de-zheng-shu-ci-fang-lcof/)

实现函数double Power(double base, int exponent)，求base的exponent次方。不得使用库函数，同时不需要考虑大数问题。

 

示例 1:

输入: 2.00000, 10
输出: 1024.00000
示例 2:

输入: 2.10000, 3
输出: 9.26100
示例 3:

输入: 2.00000, -2
输出: 0.25000
解释: 2-2 = 1/22 = 1/4 = 0.25
 

说明:

-100.0 < x < 100.0
n 是 32 位有符号整数，其数值范围是 [−231, 231 − 1] 。
注意：本题与主站 50 题相同：https://leetcode-cn.com/problems/powx-n/


```
class Solution:
    def myPow(self, x: float, n: int) -> float:
        if n < 0:  x, n = 1/x, -n
        res = 1
        while n:
            if n & 1:  res *= x  # n为奇数，多乘一个x
            x *= x 
            n >>= 1
        return res 

"""
首先明确：
x的n次方 = (x的n/2次方)的平方，并依据这点写递归调用。
然后我们考虑几种情况：
（1）偶数次幂直接用上一次递归结果half自身的平方
（2）奇数次幂，在乘方的基础上，再乘以x本身，即half*half*x
注意：奇数次幂，python的整除是向下取整，如-3//2=-2，所以(-n)//2（假设n为正整数）的绝对值会比n//2的结果大1，那么half*half就多算了一次，所以要去掉一个x的-1次幂，即乘以x，算法反而与正数一样。所以，python编写程序，奇数次幂不需要判断n的正负，其他语言整除可能是向零取整，需要判断正负.


"""
# class Solution:
#     def myPow(self, x: float, n: int) -> float:
#         if n == 0: return 1
#         if n == -1: return 1/x
#         half = self.myPow(x, n//2)
#         if n % 2 == 0:
#             return half * half
#         else:
#             return half * half * x
```
---

#### [面试题17\. 打印从1到最大的n位数](https://leetcode-cn.com/problems/da-yin-cong-1dao-zui-da-de-nwei-shu-lcof/)
输入数字 n，按顺序打印出从 1 到最大的 n 位十进制数。比如输入 3，则打印出 1、2、3 一直到最大的 3 位数 999。

示例 1:

输入: n = 1
输出: [1,2,3,4,5,6,7,8,9]
 

说明：

用返回一个整数列表来代替打印
n 为正整数


```
# 将 1 到 10^n−1 分别添加到数组
class Solution:
    def printNumbers(self, n: int) -> List[int]:
        return [i for i in range(1, 10 ** n)]
```
---
#### [面试题18\. 删除链表的节点](https://leetcode-cn.com/problems/shan-chu-lian-biao-de-jie-dian-lcof/)
给定单向链表的头指针和一个要删除的节点的值，定义一个函数删除该节点。

返回删除后的链表的头节点。

注意：此题对比原题有改动

示例 1:

输入: head = [4,5,1,9], val = 5
输出: [4,1,9]
解释: 给定你链表中值为 5 的第二个节点，那么在调用了你的函数之后，该链表应变为 4 -> 1 -> 9.
示例 2:

输入: head = [4,5,1,9], val = 1
输出: [4,5,9]
解释: 给定你链表中值为 1 的第三个节点，那么在调用了你的函数之后，该链表应变为 4 -> 5 -> 9.
 

说明：

题目保证链表中节点的值互不相同
若使用 C 或 C++ 语言，你不需要 free 或 delete 被删除的节点


```
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def deleteNode(self, head: ListNode, val: int) -> ListNode:
        dummy = ListNode(0)
        dummy.next = head
        if head.val == val: return head.next
        while head and head.next:
            if head.next.val == val:
                head.next = head.next.next
            head = head.next
        return dummy.next
```
---
#### [面试题19\. 正则表达式匹配](https://leetcode-cn.com/problems/zheng-ze-biao-da-shi-pi-pei-lcof/)
请实现一个函数用来匹配包含'. '和'*'的正则表达式。模式中的字符'.'表示任意一个字符，而'*'表示它前面的字符可以出现任意次（含0次）。在本题中，匹配是指字符串的所有字符匹配整个模式。例如，字符串"aaa"与模式"a.a"和"ab*ac*a"匹配，但与"aa.a"和"ab*a"均不匹配。

示例 1:

输入:
s = "aa"
p = "a"
输出: false
解释: "a" 无法匹配 "aa" 整个字符串。
示例 2:

输入:
s = "aa"
p = "a*"
输出: true
解释: 因为 '*' 代表可以匹配零个或多个前面的那一个元素, 在这里前面的元素就是 'a'。因此，字符串 "aa" 可被视为 'a' 重复了一次。
示例 3:

输入:
s = "ab"
p = ".*"
输出: true
解释: ".*" 表示可匹配零个或多个（'*'）任意字符（'.'）。
示例 4:

输入:
s = "aab"
p = "c*a*b"
输出: true
解释: 因为 '*' 表示零个或多个，这里 'c' 为 0 个, 'a' 被重复一次。因此可以匹配字符串 "aab"。
示例 5:

输入:
s = "mississippi"
p = "mis*is*p*."
输出: false
s 可能为空，且只包含从 a-z 的小写字母。
p 可能为空，且只包含从 a-z 的小写字母，以及字符 . 和 *。
注意：本题与主站 10 题相同：https://leetcode-cn.com/problems/regular-expression-matching/


```
"""
定义一个二维数组dp，dp[i][j]表示s的前i个字符和p的前j个字符是匹配的
dp[i][j]的计算方式如下

首先设置dp[0][0]为true，因为两个空字符是匹配的
如果i = 0, 那么表示以空字符串去匹配p的前j个字符，我们期望p[j] == , 这样之前的字符不用出现，dp[i][j] = p[j] == * and dp[i][j-2]
如果s[i] == p[j]那么，直接看i-1, 和j-1是不是匹配的，dp[i][j] = dp[i-1][j-1]
最后就是需要处理的情况，有两种选择，重复前字符一次，或者不要这个字符，只要其中一个能匹配就行
不要前一个字符， dp[i][j-2]
重复一次，需要满足条件p[j-1] == s[i] 或者p[j-1] == '.', dp[i-1][j]
最后返回dp[m][n]就是能不能匹配的结果
"""
class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        s, p = '#'+s, '#'+p
        m, n = len(s), len(p)
        dp = [[False]*n for _ in range(m)]
        dp[0][0] = True
        
        for i in range(m):
            for j in range(1, n):
                if i == 0:
                    dp[i][j] = j > 1 and p[j] == '*' and dp[i][j-2]
                elif p[j] in [s[i], '.']:
                    dp[i][j] = dp[i-1][j-1]
                elif p[j] == '*':
                    dp[i][j] = j > 1 and dp[i][j-2] or p[j-1] in [s[i], '.'] and dp[i-1][j]
                else:
                    dp[i][j] = False
        return dp[-1][-1]
```
---
#### [面试题20\. 表示数值的字符串](https://leetcode-cn.com/problems/biao-shi-shu-zhi-de-zi-fu-chuan-lcof/)
请实现一个函数用来判断字符串是否表示数值（包括整数和小数）。例如，字符串"+100"、"5e2"、"-123"、"3.1416"、"0123"及"-1E-16"都表示数值，但"12e"、"1a3.14"、"1.2.3"、"+-5"及"12e+5.4"都不是。

 

注意：本题与主站 65 题相同：https://leetcode-cn.com/problems/valid-number/


```
class Solution:
    def isNumber(self, s: str) -> bool:
        i = 0
        j = len(s) - 1
        while i < len(s) and s[i] == ' ':
            i += 1
        while j > 0 and s[j] == ' ':
            j -= 1
        if i > j:
            return False

        s = s[i:j + 1] # 关键

        if s[0] == '-' or s[0] == '+':
            s = s[1:]
        if len(s) == 0 or s[0] == '.' and len(s) == 1:
            return False

        e = 0
        dot = 0
        i = 0

        while i < len(s): # 此处不可以用for loop 因为i+=1不会改变
            if s[i] >= '0' and s[i] <= '9':
                i += 1
                continue
            elif s[i] == '.':
                dot += 1
                if e or dot > 1: # 如果e后面出现'.' 或者多个'.' False
                    return False
                i += 1
            elif s[i] == 'e' or s[i] == 'E':
                e += 1
                if i + 1 == len(s) or i == 0 or e > 1 or i == 1 and s[0] == '.': # 如果e后面没有数字或.e False
                    return False
                if s[i + 1] == '+' or s[i + 1] == '-': #如果e后面出现符号之后没有数字 False
                    if i + 2 == len(s):
                        return False
                    i += 1
                i += 1
            else:
                return False

        return True
        
        # try:
        #     float(s)
        #     return True
        # except:
        #     return False

```
---
#### [面试题21\. 调整数组顺序使奇数位于偶数前面](https://leetcode-cn.com/problems/diao-zheng-shu-zu-shun-xu-shi-qi-shu-wei-yu-ou-shu-qian-mian-lcof/)
输入一个整数数组，实现一个函数来调整该数组中数字的顺序，使得所有奇数位于数组的前半部分，所有偶数位于数组的后半部分。

 

示例：

输入：nums = [1,2,3,4]
输出：[1,3,2,4] 
注：[3,1,2,4] 也是正确的答案之一。
 

提示：

1 <= nums.length <= 50000
1 <= nums[i] <= 10000


```
class Solution:
    def exchange(self, nums: List[int]) -> List[int]:
        left, right = 0, len(nums)-1
        while left < right:
            while left < right and nums[left] % 2 == 1 : 
                left += 1
            while left < right and nums[right] % 2 == 0: 
                right -= 1
            nums[left], nums[right] = nums[right], nums[left]
        return nums

        # left, right = [], []
        # for x in nums:
        #     if x % 2:
        #         left.append(x)
        #     else:
        #         right.append(x)
        # return left + right
                
```
---
#### [面试题22\. 链表中倒数第k个节点](https://leetcode-cn.com/problems/lian-biao-zhong-dao-shu-di-kge-jie-dian-lcof/)
输入一个链表，输出该链表中倒数第k个节点。为了符合大多数人的习惯，本题从1开始计数，即链表的尾节点是倒数第1个节点。例如，一个链表有6个节点，从头节点开始，它们的值依次是1、2、3、4、5、6。这个链表的倒数第3个节点是值为4的节点。

 

示例：

给定一个链表: 1->2->3->4->5, 和 k = 2.

返回链表 4->5.


```
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def getKthFromEnd(self, head: ListNode, k: int) -> ListNode:
        fast = slow = head
        while k:
            fast = fast.next 
            k -= 1
        while fast != None:
            fast = fast.next
            slow = slow.next
        return slow

        # fast = slow = head
        # for i in range(k):
        #     fast = fast.next
        # while fast != None:
        #     fast = fast.next
        #     slow = slow.next
        # return slow



        # length = 0
        # cur = head
        # while cur:
        #     cur = cur.next
        #     length += 1
        # for i in range(length-k):
        #     head = head.next
        # return head
```
---
#### [面试题24\. 反转链表](https://leetcode-cn.com/problems/fan-zhuan-lian-biao-lcof/)
定义一个函数，输入一个链表的头节点，反转该链表并输出反转后链表的头节点。

 

示例:

输入: 1->2->3->4->5->NULL
输出: 5->4->3->2->1->NULL
 

限制：

0 <= 节点个数 <= 5000

 

注意：本题与主站 206 题相同：https://leetcode-cn.com/problems/reverse-linked-list/


```
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        pre, cur = None, head
        while cur:
            tmp = cur.next
            cur.next = pre
            pre, cur = cur, tmp
        return pre

        # if not head or not head.next: return head
        # tail = self.reverseList(head.next) #递归求解 1 -> (2->3->4->5->NULL)
        # head.next.next = head # 子问题：反转  NULL<- 1 <-(2->3->4->5->NULL)
        # head.next = None
        # return tail
```
---
#### [面试题25\. 合并两个排序的链表](https://leetcode-cn.com/problems/he-bing-liang-ge-pai-xu-de-lian-biao-lcof/)
输入两个递增排序的链表，合并这两个链表并使新链表中的节点仍然是递增排序的。

示例1：

输入：1->2->4, 1->3->4
输出：1->1->2->3->4->4
限制：

0 <= 链表长度 <= 1000

注意：本题与主站 21 题相同：https://leetcode-cn.com/problems/merge-two-sorted-lists/


```
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        dummy = cur = ListNode(0)
        while l1 and l2:
            if l1.val < l2.val:
                cur.next = l1
                l1 = l1.next
            else:
                cur.next = l2
                l2 = l2.next
            cur = cur.next
        cur.next = l1 or l2
        return dummy.next

        # if l1 is None: return l2
        # if l2 is None: return l1
        # if l1.val < l2.val:
        #     l1.next = self.mergeTwoLists(l1.next, l2)
        #     return l1
        # else:
        #     l2.next = self.mergeTwoLists(l1, l2.next)
        #     return l2
```
---
#### [面试题26\. 树的子结构](https://leetcode-cn.com/problems/shu-de-zi-jie-gou-lcof/)
输入两棵二叉树A和B，判断B是不是A的子结构。(约定空树不是任意一个树的子结构)

B是A的子结构， 即 A中有出现和B相同的结构和节点值。

例如:
给定的树 A:
```
     3
    / \
   4   5
  / \
 1   2
```
给定的树 B：
```
   4 
  /
 1
```
返回 true，因为 B 与 A 的一个子树拥有相同的结构和节点值。

示例 1：

输入：A = [1,2,3], B = [3,1]
输出：false
示例 2：

输入：A = [3,4,5,1,2], B = [4,1]
输出：true
限制：

0 <= 节点个数 <= 10000


```
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def isSubStructure(self, A: TreeNode, B: TreeNode) -> bool:
        if not A or not B : return False
        if self.isSame(A, B): return True
        return self.isSubStructure(A.left, B) or self.isSubStructure(A.right, B)
    def isSame(self, A, B):
        if not B: return True
        if not A or A.val != B.val: return False
        return self.isSame(A.left, B.left) and self.isSame(A.right, B.right)
```
---
#### [面试题27\. 二叉树的镜像](https://leetcode-cn.com/problems/er-cha-shu-de-jing-xiang-lcof/)
请完成一个函数，输入一个二叉树，该函数输出它的镜像。
```
例如输入：

     4
   /   \
  2     7
 / \   / \
1   3 6   9
镜像输出：

     4
   /   \
  7     2
 / \   / \
9   6 3   1
```
 
示例 1：

输入：root = [4,2,7,1,3,6,9]
输出：[4,7,2,9,6,3,1]
 

限制：

0 <= 节点个数 <= 1000

注意：本题与主站 226 题相同：https://leetcode-cn.com/problems/invert-binary-tree/


```
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def mirrorTree(self, root: TreeNode) -> TreeNode:
        if not root: return None
        root.left, root.right = root.right, root.left
        self.mirrorTree(root.left)
        self.mirrorTree(root.right)
        return root
```
---
#### [面试题28\. 对称的二叉树](https://leetcode-cn.com/problems/dui-cheng-de-er-cha-shu-lcof/)
请实现一个函数，用来判断一棵二叉树是不是对称的。如果一棵二叉树和它的镜像一样，那么它是对称的。
```
例如，二叉树 [1,2,2,3,4,4,3] 是对称的。

    1
   / \
  2   2
 / \ / \
3  4 4  3
但是下面这个 [1,2,2,null,3,null,3] 则不是镜像对称的:

    1
   / \
  2   2
   \   \
   3    3
```
 

示例 1：

输入：root = [1,2,2,3,4,4,3]
输出：true
示例 2：

输入：root = [1,2,2,null,3,null,3]
输出：false
 

限制：

0 <= 节点个数 <= 1000

注意：本题与主站 101 题相同：https://leetcode-cn.com/problems/symmetric-tree/


```
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def isSymmetric(self, root: TreeNode) -> bool:
        return not root or self.dfs(root.left, root.right)
    def dfs(self, p, q):
        if not p or not q: return not p and not q #都是空，返回True,否则返回False
        return p.val == q.val and self.dfs(p.left, q.right) and self.dfs(p.right, q.left)
```
---
#### [面试题29\. 顺时针打印矩阵](https://leetcode-cn.com/problems/shun-shi-zhen-da-yin-ju-zhen-lcof/)
输入一个矩阵，按照从外向里以顺时针的顺序依次打印出每一个数字。

 

示例 1：

输入：matrix = [[1,2,3],[4,5,6],[7,8,9]]
输出：[1,2,3,6,9,8,7,4,5]
示例 2：

输入：matrix = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
输出：[1,2,3,4,8,12,11,10,9,5,6,7]
 

限制：

0 <= matrix.length <= 100
0 <= matrix[i].length <= 100
注意：本题与主站 54 题相同：https://leetcode-cn.com/problems/spiral-matrix/


```
class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        if not matrix: return []
        m, n = len(matrix), len(matrix[0])
        x = y = di = 0 
        dx = [0, 1, 0, -1] #上右下左
        dy = [1, 0, -1, 0]
        res = []
        visited = set()
        for i in range(m*n):
            res.append(matrix[x][y])
            visited.add((x,y))
            nx, ny = x+dx[di], y+dy[di]
            if 0<=nx<m and 0<=ny<n and (nx,ny) not in visited:
                x, y = nx, ny
            else:
                di = (di + 1)%4
                x, y = x+dx[di], y+dy[di]
        return res

# """将矩阵第一行的元素添加到res列表里，删除第一行（也就是matrix中的第一个列表），
# 然后逆时针旋转（这里通过转置+倒序实现），新的matrix的第一行即为接下来要打印的元素。
# """"
#         res = []
#         while matrix:
#             res += matrix.pop(0)
#             matrix = list(zip(*matrix))[::-1] #逆时针旋转（转置+倒序）
#         return res
```
---























#### [面试题30\. 包含min函数的栈](https://leetcode-cn.com/problems/bao-han-minhan-shu-de-zhan-lcof/)
定义栈的数据结构，请在该类型中实现一个能够得到栈的最小元素的 min 函数在该栈中，调用 min、push 及 pop 的时间复杂度都是 O(1)。

 

示例:

MinStack minStack = new MinStack();
minStack.push(-2);
minStack.push(0);
minStack.push(-3);
minStack.min();   --> 返回 -3.
minStack.pop();
minStack.top();      --> 返回 0.
minStack.min();   --> 返回 -2.
 

提示：

各函数的调用总次数不超过 20000 次
 

注意：本题与主站 155 题相同：https://leetcode-cn.com/problems/min-stack/


```
class MinStack:

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.stack = []
        self.minstack = []

    def push(self, x: int) -> None:
        self.stack.append(x)
        if not self.minstack or x <= self.minstack[-1]: 
            self.minstack.append(x)

    def pop(self) -> None:
        tmp = self.stack.pop()
        if tmp == self.minstack[-1]: self.minstack.pop()
        
    def top(self) -> int:
        return self.stack[-1]

    def min(self) -> int:
        return self.minstack[-1]



# Your MinStack object will be instantiated and called as such:
# obj = MinStack()
# obj.push(x)
# obj.pop()
# param_3 = obj.top()
# param_4 = obj.min()
```
---
# C++
```
class MinStack {
public:
    /** initialize your data structure here. */
    stack<int> StackValue;
    stack<int> StackMin;
    MinStack() {

    }
    
    void push(int x) {
        StackValue.push(x);
        if(StackMin.empty() || StackMin.top() >= x)
            StackMin.push(x);
    }
    
    void pop() {
        if(StackValue.top() == StackMin.top()) StackMin.pop();
        StackValue.pop();
    }
    
    int top() {
        return StackValue.top();
    }
    
    int min() {
        return StackMin.top();
    }
};

/**
 * Your MinStack object will be instantiated and called as such:
 * MinStack* obj = new MinStack();
 * obj->push(x);
 * obj->pop();
 * int param_3 = obj->top();
 * int param_4 = obj->min();
 */
```
---
#### [面试题31\. 栈的压入、弹出序列](https://leetcode-cn.com/problems/zhan-de-ya-ru-dan-chu-xu-lie-lcof/)
输入两个整数序列，第一个序列表示栈的压入顺序，请判断第二个序列是否为该栈的弹出顺序。假设压入栈的所有数字均不相等。例如，序列 {1,2,3,4,5} 是某栈的压栈序列，序列 {4,5,3,2,1} 是该压栈序列对应的一个弹出序列，但 {4,3,5,1,2} 就不可能是该压栈序列的弹出序列。

 

示例 1：

输入：pushed = [1,2,3,4,5], popped = [4,5,3,2,1]
输出：true
解释：我们可以按以下顺序执行：
push(1), push(2), push(3), push(4), pop() -> 4,
push(5), pop() -> 5, pop() -> 3, pop() -> 2, pop() -> 1
示例 2：

输入：pushed = [1,2,3,4,5], popped = [4,3,5,1,2]
输出：false
解释：1 不能在 2 之前弹出。
 

提示：

0 <= pushed.length == popped.length <= 1000
0 <= pushed[i], popped[i] < 1000
pushed 是 popped 的排列。
注意：本题与主站 946 题相同：https://leetcode-cn.com/problems/validate-stack-sequences/

```
class Solution:
    def validateStackSequences(self, pushed: List[int], popped: List[int]) -> bool:
        stack = []
        j = 0
        for x in pushed:
            stack.append(x)
            while stack and j < len(popped) and stack[-1] == popped[j]:
                stack.pop()
                j += 1
        return j == len(popped)
        # return not stack
```
# C++
```
class Solution {
public:
    bool validateStackSequences(vector<int>& pushed, vector<int>& popped) {
        stack<int> stk;
        int n = popped.size();
        int j = 0;
        for(int i=0; i < pushed.size(); ++i){
            stk.push(pushed[i]);
            while(!stk.empty() && j<n && stk.top() == popped[j]){
                stk.pop();
                ++j;
            }
        }
        return stk.empty();
    }
};
```
# C++
```
class Solution {
public:
    bool validateStackSequences(vector<int>& pushed, vector<int>& popped) {
        stack<int> stk;
        int j = 0;
        for(auto x :pushed){
            stk.push(x);
            while(!stk.empty() && j < popped.size() && stk.top()== popped[j]){
                stk.pop();
                ++j;
            }
        }
        // return j == popped.size();
        return stk.empty();
    }
};
```
---

#### [面试题32 - I. 从上到下打印二叉树](https://leetcode-cn.com/problems/cong-shang-dao-xia-da-yin-er-cha-shu-lcof/)
从上到下打印出二叉树的每个节点，同一层的节点按照从左到右的顺序打印。

 
```
例如:
给定二叉树: [3,9,20,null,null,15,7],

    3
   / \
  9  20
    /  \
   15   7
返回：

[3,9,20,15,7]
 ```

提示：

节点总数 <= 1000

```
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def levelOrder(self, root: TreeNode) -> List[int]:
        if not root: return []
        res = []
        q = [root]
        while q:
            node = q.pop(0)
            res.append(node.val)
            if node.left: q.append(node.left)
            if node.right: q.append(node.right)
        return res
```
# C++
```
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    vector<int> levelOrder(TreeNode* root) {
        vector<int> res;
        if(!root) return res;
        queue<TreeNode *> q;
        q.push(root);
        while(q.size()){
            auto t = q.front();
            q.pop();
            res.push_back(t->val);
            if(t->left) q.push(t->left);
            if(t->right) q.push(t->right);
        }
        return res;
    }
};
```
---
#### [面试题32 - II. 从上到下打印二叉树 II](https://leetcode-cn.com/problems/cong-shang-dao-xia-da-yin-er-cha-shu-ii-lcof/)
从上到下按层打印二叉树，同一层的节点按从左到右的顺序打印，每一层打印到一行。

 
```
例如:
给定二叉树: [3,9,20,null,null,15,7],

    3
   / \
  9  20
    /  \
   15   7
返回其层次遍历结果：

[
  [3],
  [9,20],
  [15,7]
]
 ```

提示：

节点总数 <= 1000
注意：本题与主站 102 题相同：https://leetcode-cn.com/problems/binary-tree-level-order-traversal/

```
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        if not root: return []
        queue = [root]
        res = []
        while queue:
            tmp = []
            for _ in range(len(queue)):
                if not queue: break
                node = queue.pop(0)
                tmp.append(node.val)
                if node.left:  queue.append(node.left)
                if node.right: queue.append(node.right)
            res.append(tmp)
        return res
```
```
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        levels = []
        if not root: return levels
        def helper(node, level):
            if len(levels) == level: levels.append([])
            levels[level].append(node.val)
            if node.left:  helper(node.left, level+1)
            if node.right: helper(node.right, level+1)
        helper(root, 0)
        return levels      
```
```
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
from collections import deque
class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        levels = []
        level = 0
        if not root: return levels
        queue = deque([root,])
        while queue:
            levels.append([])
            level_length = len(queue)
            for i in range(level_length):
                node = queue.popleft()
                levels[level].append(node.val)
                if node.left:  queue.append(node.left)
                if node.right: queue.append(node.right)
            level += 1 # go to next level
        return levels
```
---
#### [面试题32 - III. 从上到下打印二叉树 III](https://leetcode-cn.com/problems/cong-shang-dao-xia-da-yin-er-cha-shu-iii-lcof/)
请实现一个函数按照之字形顺序打印二叉树，即第一行按照从左到右的顺序打印，第二层按照从右到左的顺序打印，第三行再按照从左到右的顺序打印，其他行以此类推。

 
```
例如:
给定二叉树: [3,9,20,null,null,15,7],

    3
   / \
  9  20
    /  \
   15   7
返回其层次遍历结果：

[
  [3],
  [20,9],
  [15,7]
]
 

提示：

1.节点总数 <= 1000
```

```
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        res = []
        if not root: return res
        queue = [root]
        level = 0
        while queue:
            tmp = []
            level_length = len(queue)
            for i in range(level_length):
                node = queue.pop(0)
                tmp.append(node.val)
                if node.left: queue.append(node.left)
                if node.right: queue.append(node.right)
            if level % 2 : tmp.reverse()
            res.append(tmp)
            level += 1
        return res
```
---
#### [面试题33\. 二叉搜索树的后序遍历序列](https://leetcode-cn.com/problems/er-cha-sou-suo-shu-de-hou-xu-bian-li-xu-lie-lcof/)
输入一个整数数组，判断该数组是不是某二叉搜索树的后序遍历结果。如果是则返回 true，否则返回 false。假设输入的数组的任意两个数字都互不相同。

 

参考以下这颗二叉搜索树：
```
     5
    / \
   2   6
  / \
 1   3
```
示例 1：

输入: [1,6,3,2,5]
输出: false
示例 2：

输入: [1,3,2,6,5]
输出: true
 

提示：

数组长度 <= 1000

```
class Solution:
    def verifyPostorder(self, postorder: List[int]) -> bool:
        if not postorder: return True
        i = 0
        while postorder[i] < postorder[-1]:
            i += 1
        for x in postorder[i:-1]:
            if x < postorder[-1]:
                return False
        return self.verifyPostorder(postorder[:i]) and self.verifyPostorder(postorder[i:-1])
```
---

#### [面试题34\. 二叉树中和为某一值的路径](https://leetcode-cn.com/problems/er-cha-shu-zhong-he-wei-mou-yi-zhi-de-lu-jing-lcof/)
输入一棵二叉树和一个整数，打印出二叉树中节点值的和为输入整数的所有路径。从树的根节点开始往下一直到叶节点所经过的节点形成一条路径。

 
示例:
给定如下二叉树，以及目标和 sum = 22，
```
              5
             / \
            4   8
           /   / \
          11  13  4
         /  \    / \
        7    2  5   1
```
返回:

[
   [5,4,11,2],
   [5,8,4,5]
]
 

提示：

节点总数 <= 10000
注意：本题与主站 113 题相同：https://leetcode-cn.com/problems/path-sum-ii/

## C++
```
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    vector<vector<int> > ans;
    vector<int> path;
    vector<vector<int>> pathSum(TreeNode* root, int sum) {
        dfs(root, sum);
        return ans;
    }
    void dfs(TreeNode* root, int sum){
        if(!root) return;
        path.push_back(root->val);
        sum -= root->val;
        if(!root->left && !root->right && !sum) ans.push_back(path);
        dfs(root->left, sum);
        dfs(root->right, sum);
        path.pop_back(); //删除向量中的最后一个元素，有效地将容器大小减小了一个。
    }
};
```
## Python3
```
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def pathSum(self, root: TreeNode, sum: int) -> List[List[int]]:
        ans, path = [],[]
        def dfs(root, sum):
            if not root: return
            path.append(root.val)
            sum -= root.val
            if not root.left and not root.right and not sum:
                ans.append(path[:])
            dfs(root.left, sum)
            dfs(root.right, sum)
            path.pop()
        
        dfs(root, sum)
        return ans
```
```
class Solution:
    def pathSum(self, root: TreeNode, sum: int) -> List[List[int]]:
        self.ans, self.path = [],[]    
        self.dfs(root, sum)
        return self.ans
    def dfs(self, root, sum):
        if not root: return
        self.path.append(root.val)
        sum -= root.val
        if not root.left and not root.right and not sum:
            self.ans.append(self.path[:])
        self.dfs(root.left, sum)
        self.dfs(root.right, sum)
        self.path.pop()
```

```python []
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def pathSum(self, root: TreeNode, sum: int) -> List[List[int]]:
        res, path = [], []

        def dfs(node, sum):
            #递归出口：解决子问题
            if not node: return #如果没有节点(node = None)，直接返回，不向下执行
            else:               #有节点
                path.append(node.val) #将节点值添加到path
                sum -= node.val 
            # 如果节点为叶子节点，并且 sum == 0
            if not node.left and not node.right and not sum: 
                res.append(path[:]) 

            dfs(node.left, sum) #递归处理左边
            dfs(node.right, sum) #递归处理右边
            path.pop() #处理完一个节点后，恢复初始状态，为node.left,  node.right操作

        dfs(root, sum)
        return res
```
---


#### [面试题35\. 复杂链表的复制](https://leetcode-cn.com/problems/fu-za-lian-biao-de-fu-zhi-lcof/)
请实现 copyRandomList 函数，复制一个复杂链表。在复杂链表中，每个节点除了有一个 next 指针指向下一个节点，还有一个 random 指针指向链表中的任意节点或者 null。

 **示例 1：**

![](https://upload-images.jianshu.io/upload_images/21384689-31b7baf5d223feaa.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

**输入：**head = [[7,null],[13,0],[11,4],[10,2],[1,0]]
**输出：**[[7,null],[13,0],[11,4],[10,2],[1,0]]
</pre>

**示例 2：**

![](https://upload-images.jianshu.io/upload_images/21384689-101b2e0027243799.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

**输入：**head = [[1,1],[2,1]]
**输出：**[[1,1],[2,1]]
</pre>

**示例 3：**

**![](https://upload-images.jianshu.io/upload_images/21384689-7a7e9ad11cb9c712.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)** 

**输入：**head = [[3,null],[3,0],[3,null]]
**输出：**[[3,null],[3,0],[3,null]]
</pre>

**示例 4：**

**输入：**head = []
**输出：**[]
**解释：**给定的链表为空（空指针），因此返回 null。
</pre>

**提示：**

*   `-10000 <= Node.val <= 10000`
*   `Node.random` 为空（null）或指向链表中的节点。
*   节点数目不超过 1000 。

**注意：**本题与主站 138 题相同：[https://leetcode-cn.com/problems/copy-list-with-random-pointer/](https://leetcode-cn.com/problems/copy-list-with-random-pointer/)

```
"""
# Definition for a Node.
class Node:
    def __init__(self, x: int, next: 'Node' = None, random: 'Node' = None):
        self.val = int(x)
        self.next = next
        self.random = random
"""
class Solution:
    def copyRandomList(self, head: 'Node') -> 'Node':
        if not head: return None

        ptr = head
        while ptr:
            new_node = Node(ptr.val, None, None)
            new_node.next = ptr.next
            ptr.next = new_node
            ptr = new_node.next

        ptr = head
        while ptr:
            ptr.next.random = ptr.random.next if ptr.random else None
            ptr = ptr.next.next

        dummy = head.next
        ptr_old_list = head
        ptr_new_list = head.next
        while ptr_old_list:
            ptr_old_list.next = ptr_old_list.next.next
            ptr_new_list.next = ptr_new_list.next.next if ptr_new_list.next else None
            ptr_old_list = ptr_old_list.next
            ptr_new_list = ptr_new_list.next
        return dummy

```

#### [面试题36\. 二叉搜索树与双向链表](https://leetcode-cn.com/problems/er-cha-sou-suo-shu-yu-shuang-xiang-lian-biao-lcof/)

输入一棵二叉搜索树，将该二叉搜索树转换成一个排序的循环双向链表。要求不能创建任何新的节点，只能调整树中节点指针的指向。

为了让您更好地理解问题，以下面的二叉搜索树为例：

 ![](https://upload-images.jianshu.io/upload_images/21384689-e31c7f71991ec255.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

我们希望将这个二叉搜索树转化为双向循环链表。链表中的每个节点都有一个前驱和后继指针。对于双向循环链表，第一个节点的前驱是最后一个节点，最后一个节点的后继是第一个节点。

下图展示了上面的二叉搜索树转化成的链表。“head” 表示指向链表中有最小元素的节点。
![](https://upload-images.jianshu.io/upload_images/21384689-21d4de47044bc2e6.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

特别地，我们希望可以就地完成转换操作。当转化完成以后，树中节点的左指针需要指向前驱，树中节点的右指针需要指向后继。还需要返回链表中的第一个节点的指针。

**注意：**本题与主站 426 题相同：[https://leetcode-cn.com/problems/convert-binary-search-tree-to-sorted-doubly-linked-list/](https://leetcode-cn.com/problems/convert-binary-search-tree-to-sorted-doubly-linked-list/)

**注意：**此题对比原题有改动。
```
"""
# Definition for a Node.
class Node:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
"""
class Solution:
    def treeToDoublyList(self, root: 'Node') -> 'Node':
        if not root: return root
        self.tail = self.head = None

        def dfs(root):
            if not root: return
            dfs(root.left) # 处理左半边
            if self.head is None:   #初始化，首尾指向根
                self.head = self.tail = root 
            if self.head == root:   # 第一次层
                return dfs(root.right)  # 处理右节点
            #处理子问题： 2 <-> 3 <-> 4
            self.tail.right = root  # 3->4
            root.left = self.tail   # 4->3
            self.tail = self.tail.right # 3->2
            dfs(root.right) #处理右半边

        dfs(root)
        self.tail.right = self.head
        self.head.left = self.tail
        return self.head
```









#### [面试题38\. 字符串的排列](https://leetcode-cn.com/problems/zi-fu-chuan-de-pai-lie-lcof/)
输入一个字符串，打印出该字符串中字符的所有排列。

 你可以以任意顺序返回这个字符串数组，但里面不能有重复元素。

示例:

输入：s = "abc"
输出：["abc","acb","bac","bca","cab","cba"]
 
限制：

1 <= s 的长度 <= 8

```
class Solution:
    def permutation(self, s: str) -> List[str]:
        if len(s) == 1: return [s]
        res = []
        for i, x in enumerate(s):
            n = s[:i] + s[i+1:]
            for y in self.permutation(n):
                res.append(x+y)
        return list(set(res))
```     
```
class Solution:
    def permutation(self, s: str) -> List[str]:
        length = len(s)
        if length == 1: return [s] # 边界
        else:
            res = []
            for i in range(length):
                ch = s[i]   #取出s中每一个字符
                rest = s[:i] + s[i+1:]
                for x in self.permutation(rest): #递归
                    res.append(ch + x)  #将ch 和子问题的解依次组合
        return list(set(res))
```   







#### [面试题39\. 数组中出现次数超过一半的数字](https://leetcode-cn.com/problems/shu-zu-zhong-chu-xian-ci-shu-chao-guo-yi-ban-de-shu-zi-lcof/)
数组中有一个数字出现的次数超过数组长度的一半，请找出这个数字。

 

你可以假设数组是非空的，并且给定的数组总是存在多数元素。

 

示例 1:

输入: [1, 2, 3, 2, 2, 2, 5, 4, 2]
输出: 2
 

限制：

1 <= 数组长度 <= 50000

 

注意：本题与主站 169 题相同：https://leetcode-cn.com/problems/majority-element/
```
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        """相等加1，不等减1，等于0重新赋值，抵消掉非众数"""
        count, res = 0, None
        for num in nums:
            if count == 0: res = num
            count += (1 if num == res else -1)
        return res 
```
 
```
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        count = {}
        for num in nums:
            if num in count:
                count[num] += 1
            else:
                count[num] = 1
            if count[num] > len(nums)>>1:
                return num
  ```
---

#### [面试题40\. 最小的k个数](https://leetcode-cn.com/problems/zui-xiao-de-kge-shu-lcof/)
输入整数数组 arr ，找出其中最小的 k 个数。例如，输入4、5、1、6、2、7、3、8这8个数字，则最小的4个数字是1、2、3、4。

 

示例 1：

输入：arr = [3,2,1], k = 2
输出：[1,2] 或者 [2,1]
示例 2：

输入：arr = [0,1,2,1], k = 1
输出：[0]
 

限制：

0 <= k <= arr.length <= 10000
0 <= arr[i] <= 10000


方法一：堆
思路和算法

我们用一个大根堆实时维护数组的前 k 小值。
首先将所有数插入大根堆中，维护一个长度为k的大根堆，堆长度大于k,就把堆顶的数弹出。最后将大根堆里的数存入数组返回即可。在下面的代码中，由于 C++ 语言中的堆（即优先队列）为大根堆，我们可以这么做。而 Python 语言中的堆为小根堆，因此我们要对数组中所有的数取其相反数，才能使用小根堆维护前 k 小值。

复杂度分析

时间复杂度：O(nlogk)，其中 n 是数组 arr 的长度。由于大根堆实时维护前 kk 小值，所以插入删除都是 O(logk) 的时间复杂度，最坏情况下数组里 n 个数都会插入，所以一共需要O(nlogk) 的时间复杂度。

空间复杂度：O(k)，因为大根堆里最多 k 个数。
## 方法一:大根堆
Python 小根堆加负号，变成大根堆
```
class Solution:
    def getLeastNumbers(self, arr: List[int], k: int) -> List[int]:
        if k == 0: 
            return list()

        hq = [-x for x in arr[:k]]
        heapq.heapify(hq)
        for i in range(k, len(arr)):
            if -hq[0] > arr[i]:
                heapq.heappop(hq)
                heapq.heappush(hq, -arr[i])
        res = [-x for x in hq]
        return res
        
```
Python 小根堆加负号，变成大根堆
```
class Solution:
    def getLeastNumbers(self, arr: List[int], k: int) -> List[int]:
        heap, res = [], []
        for x in arr:
            heapq.heappush(heap, -x)
            if len(heap) > k: heapq.heappop(heap)
        while len(heap): 
            res.append(-heap[0])
            heapq.heappop(heap)
        return res
```
C++大根堆
```
class Solution {
public:
    vector<int> getLeastNumbers(vector<int>& arr, int k) {
        vector<int> res;
        priority_queue<int> heap;   //大根堆
        for ( auto x : arr){
            heap.push(x);
            if (heap.size() > k) heap.pop();
        }
        while (heap.size()) res.push_back(heap.top()), heap.pop();
        return res;
    }
};
```
方法二:快排partition划分
```
class Solution:
    def partition(self, nums, l, r):
        pivot = nums[l]
        while l < r:
            while l < r and pivot <= nums[r]:
                r -= 1
            nums[l] = nums[r]
            while l < r and nums[l] <= pivot:
                l += 1
            nums[r] = nums[l]
        nums[l] = pivot
        return l 

    def random_partition(self, nums, l, r):
        import random
        i = random.randint(l, r) # l r内生成一个随机数
        nums[r], nums[i] = nums[i], nums[r] #交换
        return self.partition(nums, l, r) 

    def getLeastNumbers(self, arr: List[int], k: int) -> List[int]:
        if k == 0: return list()
        l, r = 0, len(arr)-1
        idx = self.random_partition(arr, l, r)
        while idx != k - 1:
            if idx < k-1:
                l = idx+1
                idx = self.partition(arr, l, r)
            if idx > k-1:
                r = idx-1
                idx = self.partition(arr, l, r)
        return arr[:k]
```
```
class Solution:
    def getLeastNumbers(self, arr: List[int], k: int) -> List[int]:
        """快排partition划分"""      
        if len(arr) < k or k <= 0: return []
        low, high = 0, len(arr)-1
        idx = self.partition(arr, low, high)
        while idx != k-1:
            if idx < k-1:
                low = idx + 1
                idx = self.partition(arr, low, high)
            if k-1 < idx:
                high = idx - 1
                idx = self.partition(arr, low, high)
        return arr[:k]

    def partition(self, nums, low, high):
        pivot = nums[low]
        while low < high:
            while low < high and pivot <= nums[high]:
                high -= 1 
            nums[low] = nums[high]
            while low < high and nums[low] <= pivot:
                low += 1
            nums[high] = nums[low]
        nums[low] = pivot
        return low

        # arr.sort()
        # return arr[:k]
```
---
#方法二：堆
```
def sink(array, k):
    n = len(array)
    left = 2 * k + 1
    right = 2 * k + 2
    if left >= n: return
    min_i = left 
    if right < n and array[left] > array[right]:
        min_i = right
    if array[min_i] < array[k]:
        array[min_i], array[k] = array[k], array[min_i]
        sink(array, min_i)

def build_heap(list):
    n = len(list)
    for i in range(n//2, -1, -1):
        sink(list, i)

    return list

def GetLeastNumbers_Solution(tinput, k):
    if k > len(tinput): return []
    heap = build_heap(tinput)  # 建堆o(n)复杂度
    res = []
    for _ in range(k):  # 取topk o(klogn)复杂度
        heap[0], heap[-1] = heap[-1], heap[0]
        res.append(heap.pop())
        sink(heap, 0)
    return res

```
---
方法三:排序
Python3
```
class Solution:
    def getLeastNumbers(self, arr: List[int], k: int) -> List[int]:
        # arr.sort()
        # return arr[:k]
        res = sorted(arr)
        return res[:k]
```
C++
```
class Solution {
public:
    vector<int> getLeastNumbers(vector<int>& arr, int k) {
        vector<int>vec(k,0);
        sort(arr.begin(), arr.end());
        for(int i = 0; i < k; ++i) vec[i] = arr[i];
        return vec;
    }
};
```
复杂度分析

时间复杂度：O(nlogn)，其中 n 是数组 arr 的长度。算法的时间复杂度即排序的时间复杂度。

空间复杂度：O(logn)，排序所需额外的空间复杂度为 O(logn)。












#### [面试题42\. 连续子数组的最大和](https://leetcode-cn.com/problems/lian-xu-zi-shu-zu-de-zui-da-he-lcof/)
输入一个整型数组，数组里有正数也有负数。数组中的一个或连续多个整数组成一个子数组。求所有子数组的和的最大值。

要求时间复杂度为O(n)。

 

示例1:

输入: nums = [-2,1,-3,4,-1,2,1,-5,4]
输出: 6
解释: 连续子数组 [4,-1,2,1] 的和最大，为 6。
 

提示：

1 <= arr.length <= 10^5
-100 <= arr[i] <= 100
注意：本题与主站 53 题相同：https://leetcode-cn.com/problems/maximum-subarray/
```
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        """贪心"""
        n = len(nums)
        curr_sum = max_sum = nums[0] 
        for i in range(1, n):
            curr_sum = max(nums[i], curr_sum+nums[i]) #当前元素位置的最大和
            max_sum = max(curr_sum, max_sum) #迄今为止看过的最大和
        return max_sum
```
```
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        """动态规划"""
        n = len(nums)
        max_sum = nums[0]
        for i in range(1, n):
            if nums[i-1] > 0:
                nums[i] += nums[i-1]
            max_sum = max(nums[i], max_sum) #在知道当前位置的最大和后更新全局最大和
        return max_sum
```
```
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        """动态规划"""
        res, s = -float('inf'), 0
        for x in nums:
            if s<0: s = 0
            s += x
            res = max(res, s)
        return res
```
---





















#### [面试题48\. 最长不含重复字符的子字符串](https://leetcode-cn.com/problems/zui-chang-bu-han-zhong-fu-zi-fu-de-zi-zi-fu-chuan-lcof/)
请从字符串中找出一个最长的不包含重复字符的子字符串，计算该最长子字符串的长度。

**示例 1:**
**输入:** "abcabcbb"
**输出:** 3 
**解释:** 因为无重复字符的最长子串是 `"abc"，所以其`长度为 3。
</pre>

**示例 2:**
**输入:** "bbbbb"
**输出:** 1
**解释:** 因为无重复字符的最长子串是 `"b"`，所以其长度为 1。
</pre>

**示例 3:**
**输入:** "pwwkew"
**输出:** 3
**解释:** 因为无重复字符的最长子串是 `"wke"`，所以其长度为 3。
     请注意，你的答案必须是 **子串** 的长度，`"pwke"` 是一个*子序列，*不是子串。
</pre>

提示：

*   `s.length <= 40000`

注意：本题与主站 3 题相同：[https://leetcode-cn.com/problems/longest-substring-without-repeating-characters/](https://leetcode-cn.com/problems/longest-substring-without-repeating-characters/)

### 代码

```
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        usedChar = set()
        i = j = 0
        maxLength = 0
        n = len(s)
        while i < n and j < n:
            if s[j] not in usedChar:
                usedChar.add(s[j])
                j += 1
                maxLength = max(maxLength, j - i)
            else:
                usedChar.remove(s[i])
                i += 1
        return maxLength
        

```
#### 优化
```
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        start = maxLength = 0
        usedChar = {}
        for i, c in enumerate(s):
            if c in usedChar and start <= usedChar[c]:
                start = usedChar[c] + 1
            else:
                maxLength = max(maxLength, i - start + 1)
            usedChar[c] = i 
        return maxLength
```
c++ 
```
class Solution {
public:
    int lengthOfLongestSubstring(string s) {
        unordered_map<char, int> hash;
        int res = 0;
        for (int i = 0, j = 0; i < s.size(); i ++){
            hash[s[i]] ++;  
            while (hash[s[i]] > 1)  hash[s[j ++]] --;  
            res = max(res, i - j + 1);
        }
        return res;
    }
};
```
---











#### [面试题50\. 第一个只出现一次的字符](https://leetcode-cn.com/problems/di-yi-ge-zhi-chu-xian-yi-ci-de-zi-fu-lcof/)
在字符串 s 中找出第一个只出现一次的字符。如果没有，返回一个单空格。

示例:

s = "abaccdeff"
返回 "b"

s = "" 
返回 " "
 

限制：

0 <= s 的长度 <= 50000

```
class Solution:
    def firstUniqChar(self, s: str) -> str:
        count = {}
        for c in s:
            count[c] = count.get(c, 0) + 1
        for c in s:
            if count[c] == 1: return c
        return " "
```
---
#### [面试题52\. 两个链表的第一个公共节点](https://leetcode-cn.com/problems/liang-ge-lian-biao-de-di-yi-ge-gong-gong-jie-dian-lcof/)
输入两个链表，找出它们的第一个公共节点。

如下面的两个链表：

![](https://upload-images.jianshu.io/upload_images/21384689-ec77c4e222bce4c2.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


在节点 c1 开始相交。

 

示例 1：
![](https://upload-images.jianshu.io/upload_images/21384689-9c66e67f78c170cf.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)




输入：intersectVal = 8, listA = [4,1,8,4,5], listB = [5,0,1,8,4,5], skipA = 2, skipB = 3
输出：Reference of the node with value = 8
输入解释：相交节点的值为 8 （注意，如果两个列表相交则不能为 0）。从各自的表头开始算起，链表 A 为 [4,1,8,4,5]，链表 B 为 [5,0,1,8,4,5]。在 A 中，相交节点前有 2 个节点；在 B 中，相交节点前有 3 个节点。
 

示例 2：
![](https://upload-images.jianshu.io/upload_images/21384689-8dd3a9e95cfda2ab.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)



输入：intersectVal = 2, listA = [0,9,1,2,4], listB = [3,2,4], skipA = 3, skipB = 1
输出：Reference of the node with value = 2
输入解释：相交节点的值为 2 （注意，如果两个列表相交则不能为 0）。从各自的表头开始算起，链表 A 为 [0,9,1,2,4]，链表 B 为 [3,2,4]。在 A 中，相交节点前有 3 个节点；在 B 中，相交节点前有 1 个节点。
 

示例 3：
![](https://upload-images.jianshu.io/upload_images/21384689-9999812fec37939a.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)



输入：intersectVal = 0, listA = [2,6,4], listB = [1,5], skipA = 3, skipB = 2
输出：null
输入解释：从各自的表头开始算起，链表 A 为 [2,6,4]，链表 B 为 [1,5]。由于这两个链表不相交，所以 intersectVal 必须为 0，而 skipA 和 skipB 可以是任意值。
解释：这两个链表不相交，因此返回 null。
 

注意：

如果两个链表没有交点，返回 null.
在返回结果后，两个链表仍须保持原有的结构。
可假定整个链表结构中没有循环。
程序尽量满足 O(n) 时间复杂度，且仅用 O(1) 内存。
本题与主站 160 题相同：https://leetcode-cn.com/problems/intersection-of-two-linked-lists/
#方法1：双指针
```
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        A, B = headA, headB
        while A != B:
            A = A.next if A else headB
            B = B.next if B else headA
        return A
```
#方法2：哈希
```
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        hash = {}
        while headA is not None and headB is not None:
            if headA in hash: return headA
            else:
                hash[headA] = 1
                headA = headA.next
            if headB in hash: return headB
            else:
                hash[headB] = 1
                headB = headB.next
        while headB is not None:
            if headB in hash: return headB
            else:
                hash[headB] = 1
                headB = headB.next
        while headA is not None:
            if headA in hash: return headA
            else:
                hash[headA] = 1
                headA = headA.next
        return None

```
#方法3：长的先走d步，然后一起走，直到相遇，返回地址
```
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        if not headA or not headB: return None
        distance = self.linkLength(headA) - self.linkLength(headB)
        if distance < 0: 
            return self.Go(headB, headA, -distance)
        else: 
            return self.Go(headA, headB, distance)

    def linkLength(self, head):
        length = 1
        while head.next:
            head = head.next
            length += 1
        return length
    def Go(self, headA, headB, distance):
        # headA 更长
        while distance:  # 长的链表先走distance步
            headA = headA.next
            distance -= 1
        while headA != headB:
            headA = headA.next
            headB = headB.next
        return headA       
```
---
#### [面试题53 - I. 在排序数组中查找数字 I](https://leetcode-cn.com/problems/zai-pai-xu-shu-zu-zhong-cha-zhao-shu-zi-lcof/)
统计一个数字在排序数组中出现的次数。

示例 1:

输入: nums = [5,7,7,8,8,10], target = 8
输出: 2
示例 2:

输入: nums = [5,7,7,8,8,10], target = 6
输出: 0
 
限制：

0 <= 数组长度 <= 50000

注意：本题与主站 34 题相同（仅返回值不同）：https://leetcode-cn.com/problems/find-first-and-last-position-of-element-in-sorted-array/
# 二分
时间复杂度必须是 O(log n) 级别
```
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        if not nums: return 0
        left, right = 0, len(nums)-1
        while left + 1 < right:
            mid = (left + right) >> 1
            if nums[mid] < target: 
                left = mid
            else: 
                right = mid
        count, end = 0,len(nums)-1
        if nums[left] == target:
            while left <= end and nums[left] == target :
                left += 1
                count += 1
            return count
        elif nums[right] == target:
            while right <= end and nums[right] == target:
                right += 1
                count += 1
            return count
        else: return 0
```
---
#### [面试题53 - II. 0～n-1中缺失的数字](https://leetcode-cn.com/problems/que-shi-de-shu-zi-lcof/)
一个长度为n-1的递增排序数组中的所有数字都是唯一的，并且每个数字都在范围0～n-1之内。在范围0～n-1内的n个数字中有且只有一个数字不在该数组中，请找出这个数字。

 

示例 1:

输入: [0,1,3]
输出: 2
示例 2:

输入: [0,1,2,3,4,5,6,7,9]
输出: 8
 

限制：

1 <= 数组长度 <= 10000

复杂度分析：
时间复杂度O(logN) ： 二分法为对数级别复杂度。
空间复杂度O(1) ： 几个变量使用常数大小的额外空间。
```
class Solution:
    def missingNumber(self, nums: List[int]) -> int:
        l, r = 0, len(nums)-1
        while l <= r:
            m = (l + r) // 2
            if nums[m] == m: 
                l = m + 1
            else:
                r = m - 1
        return l    # r l 相交退出

```


#### [面试题54\. 二叉搜索树的第k大节点](https://leetcode-cn.com/problems/er-cha-sou-suo-shu-de-di-kda-jie-dian-lcof/)
给定一棵二叉搜索树，请找出其中第k大的节点。

 
```
示例 1:

输入: root = [3,1,4,null,2], k = 1
   3
  / \
 1   4
  \
   2
输出: 4
示例 2:

输入: root = [5,3,6,2,4,null,null,1], k = 3
       5
      / \
     3   6
    / \
   2   4
  /
 1
输出: 4
 ```

限制：

1 ≤ k ≤ 二叉搜索树元素个数
## Python3
```python []
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def kthLargest(self, root: TreeNode, k: int) -> int:
        self.res, self.k = None, k
        def dfs(root):
            if not root: return 
            dfs(root.right)
            self.k -= 1
            if not self.k: self.res = root.val
            if self.k > 0: dfs(root.left)
        dfs(root)
        return self.res

```

## C++
```cpp []
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    int ans;
    int kthLargest(TreeNode* root, int k) {
        dfs(root, k);
        return ans;
    }
    void dfs(TreeNode* root, int &k){
        if (!root) return;
        dfs(root->right, k);
        k --;
        if(!k) ans = root->val;
        if(k > 0) dfs(root->left, k);
    }
};
```
## Python3
```python []
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def kthLargest(self, root: TreeNode, k: int) -> int:
        def inorder(root):
            return inorder(root.left) + [root.val] + inorder(root.right) if root else []
           
        res = inorder(root)
        return res[-k]      
```
```
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def kthLargest(self, root: TreeNode, k: int) -> int:
        self.res, self.k = None, k 

        def dfs(node):
            if not node: 
                return
            else:
                dfs(node.right)
                self.k -= 1
                if self.k == 0: 
                    self.res = node.val
                if self.k > 0:
                    dfs(node.left)

        dfs(root)
        return self.res

```
---





#### [面试题55 - I. 二叉树的深度](https://leetcode-cn.com/problems/er-cha-shu-de-shen-du-lcof/)
输入一棵二叉树的根节点，求该树的深度。从根节点到叶节点依次经过的节点（含根、叶节点）形成树的一条路径，最长路径的长度为树的深度。

例如：

给定二叉树 [3,9,20,null,null,15,7]，
```
    3
   / \
  9  20
    /  \
   15   7
```
返回它的最大深度 3 。

提示：

节点总数 <= 10000
注意：本题与主站 104 题相同：https://leetcode-cn.com/problems/maximum-depth-of-binary-tree/

## C ++
```c++ []
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    int maxDepth(TreeNode* root) {
        if (!root) return 0;
        return max(maxDepth(root->left), maxDepth(root->right)) + 1;

    }
};
```
## Python3

``` python []
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def maxDepth(self, root: TreeNode) -> int:
        if not root: return 0
        return max(self.maxDepth(root.left), self.maxDepth(root.right)) + 1
```
```python []
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
# 时间复杂度：O(N)
# 空间复杂度：O(log(N))

class Solution:
    def maxDepth(self, root: TreeNode) -> int:
        if root is None:
            return 0
        else:
            left_height = self.maxDepth(root.left)
            right_height = self.maxDepth(root.right)
        return max(left_height, right_height) + 1 # 1为根节点

```

```python []
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
# 时间复杂度：O(N)
# 空间复杂度：O(N)
class Solution:
    def maxDepth(self, root: TreeNode) -> int:
        stack = []
        if root is not None:
            stack.append((1,root))

        depth = 0
        while stack != []:
            current_depth, root = stack.pop()
            if root is not None:
                depth = max(depth, current_depth)
                stack.append((current_depth + 1, root.left))
                stack.append((current_depth + 1, root.right))
        
        return depth
```
---

#### [面试题55 - II. 平衡二叉树](https://leetcode-cn.com/problems/ping-heng-er-cha-shu-lcof/)
输入一棵二叉树的根节点，判断该树是不是平衡二叉树。如果某二叉树中任意节点的左右子树的深度相差不超过1，那么它就是一棵平衡二叉树。

 
```
示例 1:

给定二叉树 [3,9,20,null,null,15,7]

    3
   / \
  9  20
    /  \
   15   7
返回 true 。

示例 2:

给定二叉树 [1,2,2,3,3,null,null,4,4]

       1
      / \
     2   2
    / \
   3   3
  / \
 4   4
返回 false 。
```
 

限制：

1 <= 树的结点个数 <= 10000
注意：本题与主站 110 题相同：https://leetcode-cn.com/problems/balanced-binary-tree/

### 解题思路
**从底至顶（提前阻断法）**
**对二叉树做深度优先遍历DFS，递归过程中：
终止条件：当DFS越过叶子节点时，返回高度0；
返回值：**
从底至顶，返回以每个节点root为根节点的子树最大高度(左右子树中最大的高度值加1max(left,right) + 1)；
当我们发现有一例 左/右子树高度差 ＞ 1 的情况时，代表此树不是平衡树，返回-1；
当发现不是平衡树时，后面的高度计算都没有意义了，因此一路返回-1，避免后续多余计算。
最差情况是对树做一遍完整DFS，时间复杂度为 O(N)。
感谢@Krahets的分享

![](https://upload-images.jianshu.io/upload_images/21384689-2af8c54de2d12f41.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

### Python3
```python []
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def isBalanced(self, root: TreeNode) -> bool:
        return self.depth(root) != -1
    
    def depth(self, root):
        if not root: return 0
        left = self.depth(root.left)
        if left == -1: return -1
        right = self.depth(root.right)
        if right == -1: return -1
        return max(left, right)+ 1 if abs(left - right) < 2 else -1
```
### C++ 
```cpp []
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    bool isBalanced(TreeNode* root) {
        return depth(root) != -1;
    }
    int depth(TreeNode* root){
        if(!root) return 0;
        int left = depth(root->left);
        if (left == -1) return -1;
        int right = depth(root->right);
        if (right == -1) return -1;
        return  abs(left - right) < 2 ? max(left, right) + 1 : -1;
    }
};

```

---
### Python3
```python []
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
# O(n)
class Solution:
    def isBalanced(self, root: TreeNode) -> bool:
        self.res = True
        self.depth(root)
        return self.res 
        
    def depth(self, root):
        if not root: return 0
        left_depth = self.depth(root.left)
        right_depth = self.depth(root.right)
        if abs(left_depth - right_depth) > 1:
            self.res = False
        return max(left_depth, right_depth) + 1
```
### C++ 
```cpp []
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    bool ans = true;
    bool isBalanced(TreeNode* root) {
        dfs(root);
        return ans;
    }
    int dfs(TreeNode* root){
        if(!root) return 0;
        int left = dfs(root->left), right = dfs(root->right);
        if(abs(left - right) > 1) ans = false;
        return max(left, right) + 1;
    }
};
```
---

#### [面试题56 - I. 数组中数字出现的次数](https://leetcode-cn.com/problems/shu-zu-zhong-shu-zi-chu-xian-de-ci-shu-lcof/)
一个整型数组 nums 里除两个数字之外，其他数字都出现了两次。请写程序找出这两个只出现一次的数字。要求时间复杂度是O(n)，空间复杂度是O(1)。

 

示例 1：

输入：nums = [4,1,4,6]
输出：[1,6] 或 [6,1]
示例 2：

输入：nums = [1,2,10,4,1,4,3,3]
输出：[2,10] 或 [10,2]
 

限制：

2 <= nums <= 10000
```
class Solution:
    def singleNumbers(self, nums: List[int]) -> List[int]:
        xor = 0
        for x in nums:
            xor ^= x
        nums1, nums2 = 0, 0
        mark = 1
        while xor & mark == 0:
            mark <<= 1
        for num in nums:
            if num & mark == 0:
                nums1 ^= num 
            else:
                nums2 ^= num 
        return [nums1, nums2]
```
---
#### [面试题56 - II. 数组中数字出现的次数 II](https://leetcode-cn.com/problems/shu-zu-zhong-shu-zi-chu-xian-de-ci-shu-ii-lcof/)
在一个数组 nums 中除一个数字只出现一次之外，其他数字都出现了三次。请找出那个只出现一次的数字。

 

示例 1：

输入：nums = [3,4,3,3]
输出：4
示例 2：

输入：nums = [9,1,7,9,7,9,7]
输出：1
 

限制：

1 <= nums.length <= 10000
1 <= nums[i] < 2^31

##二进制下不考虑进位的加法：
本题为 136. Single Number 的拓展，136 题中我们用到了异或运算。实际上，异或运算的含义是二进制下不考虑进位的加法，即：0 xor 0=0+0=0， 0xor0=0+0=0, 0 xor 1=0+1=1 0xor1=0+1=1, 1 xor 0=1+0=11xor0=1+0=1, 1 xor 1=1+1=01xor1=1+1=0（不进位）。
## 三进制下不考虑进位的加法：
通过定义某种运算 #，使得 0 # 1 = 1，1 # 1 = 2，2 # 1 = 0。在此运算规则下，出现了 33 次的数字的二进制所有位全部抵消为 00，而留下只出现 11 次的数字二进制对应位为 11。因此，在此运算规则下将整个 arr 中数字遍历加和，留下来的结果则为只出现 11 次的数字。

#方法：位运算符：NOT，AND 和 XOR
思路

使用位运算符可以实现O(1) 的空间复杂度。

∼ x表示位运算 NOT

x & y表示位运算 AND

x ^ y表示位运算 XOR

XOR 该运算符用于检测出现奇数次的位：1、3、5 等。
# 0 与任何数 XOR 结果为该数。
## 0 ^ x=x
# 两个相同的数 XOR 结果为 0。
## x  ^ x = 0


以此类推，只有某个位置的数字出现奇数次时，该位的掩码才不为 0。

![](https://upload-images.jianshu.io/upload_images/21384689-73884a309b26501a.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

因此，可以检测出出现一次的位和出现三次的位，但是要注意区分这两种情况。

AND 和 NOT

为了区分出现一次的数字和出现三次的数字，使用两个位掩码：seen_once 和 seen_twice。

思路是：

# 仅当 seen_twice 未变时，改变 seen_once。

# 仅当 seen_once 未变时，改变seen_twice。

![](https://upload-images.jianshu.io/upload_images/21384689-cc2d24ed3453a6fe.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
位掩码 seen_once 仅保留出现一次的数字，不保留出现三次的数字。
```
# 时间复杂度：O(N)，遍历输入数组。
# 空间复杂度：O(1)，不使用额外空间。
# 仅当 seen_twice 未变时，改变 seen_once。
# 仅当 seen_once 未变时，改变seen_twice。
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        seen_once, seen_twice = 0, 0
        for num in nums: 
            seen_once  = ~ seen_twice & (seen_once ^ num)
            seen_twice = ~ seen_once  & (seen_twice ^ num)
        return seen_once
```

```
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        ones, twos = 0, 0
        for num in nums:
            ones = ones ^ num & ~twos
            twos = twos ^ num & ~ones
        return ones

```
---

#### [面试题57\. 和为s的两个数字](https://leetcode-cn.com/problems/he-wei-sde-liang-ge-shu-zi-lcof/)
输入一个递增排序的数组和一个数字s，在数组中查找两个数，使得它们的和正好是s。如果有多对数字的和等于s，则输出任意一对即可。

 

示例 1：

输入：nums = [2,7,11,15], target = 9
输出：[2,7] 或者 [7,2]
示例 2：

输入：nums = [10,26,30,31,47,60], target = 40
输出：[10,30] 或者 [30,10]
 

限制：

1 <= nums.length <= 10^5
1 <= nums[i] <= 10^6
#双指针
```
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        left, right = 0, len(nums)-1
        while left < right:
            if nums[left] + nums[right] < target:
                left += 1
            elif nums[left] + nums[right] > target:
                right -= 1
            else:
                return [nums[left], nums[right]]
```
#暴力超时
```
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        for i in range(n:=len(nums)):
            for j in range(i):
                if nums[i] + nums[j] == target:
                    return [nums[i], nums[j]]
```
#hash优化
```
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        visited = set()
        for num in nums:
            if target - num in visited:
                return [num, target - num]
            visited.add(num)
        return []
```
---
#### [面试题57 - II. 和为s的连续正数序列](https://leetcode-cn.com/problems/he-wei-sde-lian-xu-zheng-shu-xu-lie-lcof/)
输入一个正整数 target ，输出所有和为 target 的连续正整数序列（至少含有两个数）。

序列内的数字由小到大排列，不同序列按照首个数字从小到大排列。

 

示例 1：

输入：target = 9
输出：[[2,3,4],[4,5]]
示例 2：

输入：target = 15
输出：[[1,2,3,4,5],[4,5,6],[7,8]]
 

限制：

1 <= target <= 10^5
```
class Solution:
    def findContinuousSequence(self, target: int) -> List[List[int]]:
        l, r, res = 1, 2, []
        while r <= target//2 +1:
            cur_sum = sum(list(range(l, r+1)))
            if cur_sum < target:
                r += 1
            elif cur_sum > target:
                l += 1
            else:
                res.append(list(range(l, r+1)))
                r += 1
        return res

```
---
#### [面试题58 - I. 翻转单词顺序](https://leetcode-cn.com/problems/fan-zhuan-dan-ci-shun-xu-lcof/)
输入一个英文句子，翻转句子中单词的顺序，但单词内字符的顺序不变。为简单起见，标点符号和普通字母一样处理。例如输入字符串"I am a student. "，则输出"student. a am I"。

**示例 1：**
**输入:** "`the sky is blue`"
**输出: **"`blue is sky the`"
</pre>

**示例 2：**
**输入:** "  hello world!  "
**输出: **"world! hello"
**解释:** 输入字符串可以在前面或者后面包含多余的空格，但是反转后的字符不能包括。
</pre>

**示例 3：**
**输入:** "a good   example"
**输出: **"example good a"
**解释:** 如果两个单词间有多余的空格，将反转后单词间的空格减少到只含一个。
</pre>

**说明：**

*   无空格字符构成一个单词。
*   输入字符串可以在前面或者后面包含多余的空格，但是反转后的字符不能包括。
*   如果两个单词间有多余的空格，将反转后单词间的空格减少到只含一个。

**注意：**本题与主站 151 题相同：[https://leetcode-cn.com/problems/reverse-words-in-a-string/](https://leetcode-cn.com/problems/reverse-words-in-a-string/)

**注意：**此题对比原题有改动

### 解题思路

输入: "the sky is blue"
输出: "blue is sky the"

####  方法一：
**0.按空格分割成单词列表；
1.末位单词+空格 
2.最后一个单词末位不加空格**

```python []
class Solution:
    def reverseWords(self, s: str) -> str:
        """按空格分割成单词列表，1.末位单词+空格 2.最后一个单词末位不加空格"""
        if s == [] : return ""
        ls = s.split()  # ['the', 'sky', 'is', 'blue']
        if ls == []: return ""
        res = ""
        for i in range(len(ls)-1): # 少一个，处理末位不加空格
            res += ls[len(ls) - 1 - i] + " "  # 最后一个单词 + 空格  
        res += ls[0] # 末位不加空格
        return res
```

```python3 []
class Solution:
    def reverseWords(self, s: str) -> str:
        return " ".join(s.split()[::-1])
```

####  方法二：
**1.反转每个单词：eht yks si eulb
2.反转整个字符：blue is sky the**

### 代码



```cpp []
class Solution {
public:
    string reverseWords(string s) {
        int k = 0;
        for (int i = 0; i < s.size(); ++ i){
            while (i < s.size() && s[i] == ' ') ++i;  //找到第一个非空格字符
            if (i == s.size()) break;
            int j = i;
            while (j < s.size() && s[j] != ' ') ++j;    //遍历1个非空单词
            reverse(s.begin() + i, s.begin() + j);      //反转1个单词
            if (k) s[k++] = ' ';
            while (i < j) s[k++] = s[i++];      //反转后的1个单词赋给s[k]
        }
        s.erase(s.begin() + k, s.end());   //删除 k后面空格
        reverse(s.begin(), s.end());
        return s;
    }
};
```
---


#### [面试题58 - II. 左旋转字符串](https://leetcode-cn.com/problems/zuo-xuan-zhuan-zi-fu-chuan-lcof/)
字符串的左旋转操作是把字符串前面的若干个字符转移到字符串的尾部。请定义一个函数实现字符串左旋转操作的功能。比如，输入字符串"abcdefg"和数字2，该函数将返回左旋转两位得到的结果"cdefgab"。

 

示例 1：

输入: s = "abcdefg", k = 2
输出: "cdefgab"
示例 2：

输入: s = "lrloseumgh", k = 6
输出: "umghlrlose"
 

限制：

1 <= k < s.length <= 10000
```
class Solution:
    def reverseLeftWords(self, s: str, n: int) -> str:

        def reverse( start, end):
            while start < end:
                s[start], s[end] = s[end], s[start]
                start += 1
                end -= 1
                
        s = list(s)
        length = len(s) - 1
        reverse(0, n-1)
        reverse(n, length)
        reverse(0, length)
        
        return ''.join(s) 

```
---


#### [面试题59 - II. 队列的最大值](https://leetcode-cn.com/problems/dui-lie-de-zui-da-zhi-lcof/)
请定义一个队列并实现函数 max_value 得到队列里的最大值，要求函数max_value、push_back 和 pop_front 的均摊时间复杂度都是O(1)。

若队列为空，pop_front 和 max_value 需要返回 -1

示例 1：

输入: 
["MaxQueue","push_back","push_back","max_value","pop_front","max_value"]
[[],[1],[2],[],[],[]]
输出: [null,null,null,2,1,2]
示例 2：

输入: 
["MaxQueue","pop_front","max_value"]
[[],[],[]]
输出: [null,-1,-1]
 

限制：

1 <= push_back,pop_front,max_value的总操作数 <= 10000
1 <= value <= 10^5

```
class MaxQueue:
    
    def __init__(self):
        self.queue = []
        self.max_deque = [] #单调递减

    def max_value(self) -> int:
        return self.max_deque[0] if self.max_deque else -1

    def push_back(self, value: int) -> None:
        while self.max_deque and self.max_deque[-1] < value:
            self.max_deque.pop()
        self.max_deque.append(value)
        self.queue.append(value)

    def pop_front(self) -> int:
        if not self.max_deque: return -1
        res = self.queue.pop(0)
        if res == self.max_deque[0]:
            self.max_deque.pop(0)
        return res




# Your MaxQueue object will be instantiated and called as such:
# obj = MaxQueue()
# param_1 = obj.max_value()
# obj.push_back(value)
# param_3 = obj.pop_front()
```
```
import queue
class MaxQueue:
    """队列+双端队列"""

    def __init__(self):
        self.queue = queue.Queue()
        self.max_deque = queue.deque()

    def max_value(self) -> int:
        return self.max_deque[0] if self.max_deque else -1

    def push_back(self, value: int) -> None:
        while self.max_deque and self.max_deque[-1] < value:
            self.max_deque.pop()
        self.max_deque.append(value)
        self.queue.put(value)

    def pop_front(self) -> int:
        if not self.max_deque: return -1
        res = self.queue.get()
        if res == self.max_deque[0]:
            self.max_deque.popleft()
        return res
    
# Your MaxQueue object will be instantiated and called as such:
# obj = MaxQueue()
# param_1 = obj.max_value()
# obj.push_back(value)
# param_3 = obj.pop_front()

```
---


#### [面试题59 - I. 滑动窗口的最大值](https://leetcode-cn.com/problems/hua-dong-chuang-kou-de-zui-da-zhi-lcof/)
给定一个数组 `nums` 和滑动窗口的大小 `k`，请找出所有滑动窗口里的最大值。

**示例:**

**输入:** *nums* = `[1,3,-1,-3,5,3,6,7]`, 和 *k* = 3
**输出:** `[3,3,5,5,6,7] 
**解释:**` 
  滑动窗口的位置                最大值
---------------               -----
[1  3  -1] -3  5  3  6  7       3
 1 [3  -1  -3] 5  3  6  7       3
 1  3 [-1  -3  5] 3  6  7       5
 1  3  -1 [-3  5  3] 6  7       5
 1  3  -1  -3 [5  3  6] 7       6
 1  3  -1  -3  5 [3  6  7]      7

**提示：**

你可以假设 *k *总是有效的，在输入数组不为空的情况下，1 ≤ k ≤ 输入数组的大小。

注意：本题与主站 239 题相同：[https://leetcode-cn.com/problems/sliding-window-maximum/](https://leetcode-cn.com/problems/sliding-window-maximum/)

```
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        deque = collections.deque() #存储 nums 索引
        res = []
        for i in range(len(nums)):
            if deque and i - deque[0] > k -1:  #队列元素控制在 k 内
                deque.popleft()
            while deque and nums[deque[-1]] < nums[i]:  #维护一个单调递减队列
                deque.pop()
            deque.append(i)
            if i >= k - 1:
                res.append(nums[deque[0]])
        return res

```

#### [面试题62\. 圆圈中最后剩下的数字](https://leetcode-cn.com/problems/yuan-quan-zhong-zui-hou-sheng-xia-de-shu-zi-lcof/)
0,1,,n-1这n个数字排成一个圆圈，从数字0开始，每次从这个圆圈里删除第m个数字。求出这个圆圈里剩下的最后一个数字。

例如，0、1、2、3、4这5个数字组成一个圆圈，从数字0开始每次删除第3个数字，则删除的前4个数字依次是2、0、4、1，因此最后剩下的数字是3。

 

示例 1：

输入: n = 5, m = 3
输出: 3
示例 2：

输入: n = 10, m = 17
输出: 2
 

限制：

1 <= n <= 10^5
1 <= m <= 10^6

Python
```
class Solution:
    def lastRemaining(self, n: int, m: int) -> int:
        flag = 0
        for i in range(2,n+1):
            flag = (flag + m) % i 
        return flag
```

```
class Solution:
    def lastRemaining(self, n: int, m: int) -> int:
        f = 0 
        for i in range(2, n+1):
            f = (f + m) % i 
        return f 
```
复杂度分析

时间复杂度：O(n)，需要求解的函数值有 n 个。

空间复杂度：O(1)，只使用常数个变量。
```
class Solution:
    def lastRemaining(self, n: int, m: int) -> int:
        if n == 1: return 0 
        return (self.lastRemaining(n-1, m) + m) % n
```
C++
```
class Solution {
public:
    int lastRemaining(int n, int m) {
        int f = 0;
        for (int i = 2; i != n+1; ++i)
            f = (f + m) % i ;
        return f;

    }
};
```
```
class Solution {
public:
    int lastRemaining(int n, int m) {
        int flag = 0;
        for (int i = 2; i <= n; i ++)
            flag = (flag + m ) % i ;
        return flag;
    }
};
```
```
class Solution {
public:
    int lastRemaining(int n, int m) {
        if (n == 1) return 0;
        return (lastRemaining(n-1, m) + m) % n;
    }
};
```

复杂度分析

时间复杂度：O(n)，需要求解的函数值有 n 个。

空间复杂度：O(n)，函数的递归深度为 n，需要使用 O(n) 的栈空间。
---




#### [面试题63\. 股票的最大利润](https://leetcode-cn.com/problems/gu-piao-de-zui-da-li-run-lcof/)
假设把某股票的价格按照时间先后顺序存储在数组中，请问买卖该股票一次可能获得的最大利润是多少？

**示例 1:**
**输入:** [7,1,5,3,6,4]
**输出:** 5
**解释:** 在第 2 天（股票价格 = 1）的时候买入，在第 5 天（股票价格 = 6）的时候卖出，最大利润 = 6-1 = 5 。
     注意利润不能是 7-1 = 6, 因为卖出价格需要大于买入价格。
</pre>

**示例 2:**
**输入:** [7,6,4,3,1]
**输出:** 0
**解释:** 在这种情况下, 没有交易完成, 所以最大利润为 0。</pre>

**限制：**

`0 <= 数组长度 <= 10^5`

**注意：**本题与主站 121 题相同：[https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock/](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock/)
#### 方法：一次遍历

**算法**

假设给定的数组为：`[7, 1, 5, 3, 6, 4]`

如果我们在图表上绘制给定数组中的数字，我们将会得到：

![](https://upload-images.jianshu.io/upload_images/21384689-357a61671a4690d7?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

我们来假设自己来购买股票。随着时间的推移，每天我们都可以选择出售股票与否。那么，假设在第 `i` 天，如果我们要在今天卖股票，那么我们能赚多少钱呢？

显然，如果我们真的在买卖股票，我们肯定会想：如果我是在历史最低点买的股票就好了！太好了，在题目中，我们只要用一个变量记录一个历史最低价格 `minprice`，我们就可以假设自己的股票是在那天买的。那么我们在第 `i` 天卖出股票能得到的利润就是 `prices[i] - minprice`。

因此，我们只需要遍历价格数组一遍，记录历史最低点，然后在每一天考虑这么一个问题：如果我是在历史最低点买进的，那么我今天卖出能赚多少钱？当考虑完所有天数之时，我们就得到了最好的答案。


### 代码
时间复杂度：O(n) 
空间复杂度：O(1)
```
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        if len(prices) < 2: return 0
        min_price = prices[0]
        max_profit = 0
        for price in prices:
            if price < min_price:
                min_price = price
            if price - min_price > max_profit:
                max_profit = price - min_price
        return max_profit
```
#### dp
时间复杂度：O(n) 
空间复杂度：O(n)
``` 
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        if len(prices) < 2: return 0
        min_price = prices[0]
        dp = [0] * len(prices)
        for i in range(len(prices)):
            # 第i天的最大利润= max（前一天的最大利润，今天可能的最大利润）
            dp[i] = max(dp[i-1], prices[i] - min_price)
            min_price = min(min_price, prices[i] )
        return dp[-1]
```
#### 优化空间复杂度
时间复杂度：O(n) 
空间复杂度：O(1)
```
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        if len(prices) < 2: return 0
        min_price = prices[0]
        max_profit = 0
        for i in range(len(prices)):
            max_profit = max(max_profit, prices[i] - min_price)
            min_price = min(min_price, prices[i] )
        return max_profit
```




#### [面试题64\. 求1+2+…+n](https://leetcode-cn.com/problems/qiu-12n-lcof/)
求 1+2+...+n ，要求不能使用乘除法、for、while、if、else、switch、case等关键字及条件判断语句（A?B:C）。

 

示例 1：

输入: n = 3
输出: 6
示例 2：

输入: n = 9
输出: 45
 

限制：

1 <= n <= 10000

```
class Solution:
    def sumNums(self, n: int) -> int:

        return n and n + self.sumNums(n-1)
        # 如果 x 为 False，x and y 返回 False，否则它返回 y 的计算值。
```










 #### [面试题68 - I. 二叉搜索树的最近公共祖先](https://leetcode-cn.com/problems/er-cha-sou-suo-shu-de-zui-jin-gong-gong-zu-xian-lcof/)
给定一个二叉搜索树, 找到该树中两个指定节点的最近公共祖先。

百度百科中最近公共祖先的定义为：“对于有根树 T 的两个结点 p、q，最近公共祖先表示为一个结点 x，满足 x 是 p、q 的祖先且 x 的深度尽可能大（一个节点也可以是它自己的祖先）。”

例如，给定如下二叉搜索树:  root = [6,2,8,0,4,7,9,null,null,3,5]


![](https://upload-images.jianshu.io/upload_images/21384689-d037e32037d4a903.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

 
示例 1:

输入: root = [6,2,8,0,4,7,9,null,null,3,5], p = 2, q = 8
输出: 6 
解释: 节点 2 和节点 8 的最近公共祖先是 6。
示例 2:

输入: root = [6,2,8,0,4,7,9,null,null,3,5], p = 2, q = 4
输出: 2
解释: 节点 2 和节点 4 的最近公共祖先是 2, 因为根据定义最近公共祖先节点可以为节点本身。
 

说明:

所有节点的值都是唯一的。
p、q 为不同节点且均存在于给定的二叉搜索树中。
注意：本题与主站 235 题相同：https://leetcode-cn.com/problems/lowest-common-ancestor-of-a-binary-search-tree/

```python []
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        parent_val = root.val
        p_val, q_val = p.val, q.val
        if p_val < parent_val and q_val < parent_val:
            return self.lowestCommonAncestor(root.left, p, q)
        elif p_val > parent_val and q_val > parent_val:
            return self.lowestCommonAncestor(root.right, p, q)
        else: 
            return root
```

```python []
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        pointer = root
        while pointer:
            if p.val < pointer.val and q.val < pointer.val:
                pointer = pointer.left
            elif p.val > pointer.val and q.val > pointer.val:
                pointer = pointer.right
            else:
                return pointer

```

```
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        p_val, q_val = p.val, q.val
        node = root
        while node:
            parent_val = node.val
            if p_val < parent_val and q_val < parent_val:
                node = node.left
            elif p_val > parent_val and q_val > parent_val:
                node = node.right
            else:
                return node
```





#### [面试题68 - II. 二叉树的最近公共祖先](https://leetcode-cn.com/problems/er-cha-shu-de-zui-jin-gong-gong-zu-xian-lcof/)
给定一个二叉树, 找到该树中两个指定节点的最近公共祖先。

百度百科中最近公共祖先的定义为：“对于有根树 T 的两个结点 p、q，最近公共祖先表示为一个结点 x，满足 x 是 p、q 的祖先且 x 的深度尽可能大（一个节点也可以是它自己的祖先）。”

例如，给定如下二叉树:  root = [3,5,1,6,2,0,8,null,null,7,4]



 ![](https://upload-images.jianshu.io/upload_images/21384689-007edfc5ef5cdad0.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


示例 1:

输入: root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 1
输出: 3
解释: 节点 5 和节点 1 的最近公共祖先是节点 3。
示例 2:

输入: root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 4
输出: 5
解释: 节点 5 和节点 4 的最近公共祖先是节点 5。因为根据定义最近公共祖先节点可以为节点本身。
 

说明:

所有节点的值都是唯一的。
p、q 为不同节点且均存在于给定的二叉树中。
注意：本题与主站 236 题相同：https://leetcode-cn.com/problems/lowest-common-ancestor-of-a-binary-tree/

### C++
```cpp []
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        if(!root || root == p || root == q) return root;
        auto left = lowestCommonAncestor(root->left, p, q);
        auto right = lowestCommonAncestor(root->right, p, q);
        if(!left) return right;
        if(!right) return left;
        return root;      
    }
};
```
```
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        //（1）如果都不包含，return null,如果只包含p,return p,如果只包含q, return q
        if(!root || root == p || root == q) return root;
        //（2）否则root存在以root 为根的子树中既不包含p,也不包含q,分别看左右子树的返回值
        auto left = lowestCommonAncestor(root->left, p, q);
        auto right = lowestCommonAncestor(root->right, p, q);
        //左边为空：1.右边也都不包含，right = null,return null
        //        2.右边只包含p或q, right = p或q,return p或q
        //        3.右边同时包含p q, right 是最近公共祖先，最终返回最近公共祖先
        //结论：返回值和右边相同
        if(!left) return right;
        if(!right) return left;
        //（3）如果以root 为根的子树中包含p,q,return root
        return root;
        
    }
};
```
### Python3
```python []
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def lowestCommonAncestor(self, root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:
        if not root or root.val == p.val or root.val == q.val: return root
        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)
        return root if left and right else left or right
```
---
