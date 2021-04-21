
[8. 字符串转换整数 (atoi)](https://leetcode-cn.com/problems/string-to-integer-atoi/) 	next

[718. 最长重复子数组](https://leetcode-cn.com/problems/maximum-length-of-repeated-subarray/)

[144. 二叉树的前序遍历](https://leetcode-cn.com/problems/binary-tree-preorder-traversal/)

[169. 多数元素](https://leetcode-cn.com/problems/majority-element/)

[113. 路径总和 II](https://leetcode-cn.com/problems/path-sum-ii/)

[234. 回文链表](https://leetcode-cn.com/problems/palindrome-linked-list/)

[876. 链表的中间结点](https://leetcode-cn.com/problems/middle-of-the-linked-list/) 补充

[155. 最小栈](https://leetcode-cn.com/problems/min-stack/)

[56. 合并区间](https://leetcode-cn.com/problems/merge-intervals/)

[105. 从前序与中序遍历序列构造二叉树](https://leetcode-cn.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/)

[124. 二叉树中的最大路径和](https://leetcode-cn.com/problems/binary-tree-maximum-path-sum/)


------

[8. 字符串转换整数 (atoi)](https://leetcode-cn.com/problems/string-to-integer-atoi/)


[718. 最长重复子数组](https://leetcode-cn.com/problems/maximum-length-of-repeated-subarray/)


### 方法一：暴力


```go
func findLength(A []int, B []int) int {
	m, n, res := len(A), len(B), 0
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			if A[i] == B[j] {
				subLen := 1
				for i+subLen < m && j+subLen < n && A[i+subLen] == B[j+subLen] {
					subLen++
				}
				if res < subLen {
					res = subLen
				}
			}
		}
	}
	return res
}
```

Time Limit Exceeded
53/54 cases passed (N/A)




### 方法二：动态规划


#### 动态规划法：
- A 、B数组各抽出一个子数组，单看它们的末尾项，如果它们俩不一样，则公共子数组肯定不包括它们俩。
- 如果它们俩一样，则要考虑它们俩前面的子数组「能为它们俩提供多大的公共长度」。
	- 如果它们俩的前缀数组的「末尾项」不相同，由于子数组的连续性，前缀数组不能为它们俩提供公共长度
	- 如果它们俩的前缀数组的「末尾项」相同，则可以为它们俩提供公共长度：
	至于提供多长的公共长度？这又取决于前缀数组的末尾项是否相同……
#### 加上注释再讲一遍
- A 、B数组各抽出一个子数组，单看它们的末尾项，如果它们俩不一样——以它们俩为末尾项形成的公共子数组的长度为0：dp[i][j] = 0
- 如果它们俩一样，以它们俩为末尾项的公共子数组，长度保底为1——dp[i][j]至少为 1，要考虑它们俩的前缀数组——dp[i-1][j-1]能为它们俩提供多大的公共长度
1. 如果它们俩的前缀数组的「末尾项」不相同，前缀数组提供的公共长度为 0——dp[i-1][j-1] = 0
	- 以它们俩为末尾项的公共子数组的长度——dp[i][j] = 1
2. 如果它们俩的前缀数组的「末尾项」相同
	- 前缀部分能提供的公共长度——dp[i-1][j-1]，它至少为 1
	- 以它们俩为末尾项的公共子数组的长度 dp[i][j] = dp[i-1][j-1] + 1
- 题目求：最长公共子数组的长度。不同的公共子数组的末尾项不一样。我们考察不同末尾项的公共子数组，找出最长的那个。（注意下图的最下方的一句话）



![1.png](http://ww1.sinaimg.cn/large/007daNw2ly1gpfh1ulfu2j31go0wkn6i.jpg)

#### 状态转移方程
- dp[i][j] ：长度为i，末尾项为A[i-1]的子数组，与长度为j，末尾项为B[j-1]的子数组，二者的最大公共后缀子数组长度。
	如果 A[i-1] != B[j-1]， 有 dp[i][j] = 0
	如果 A[i-1] == B[j-1] ， 有 dp[i][j] = dp[i-1][j-1] + 1
- base case：如果i==0||j==0，则二者没有公共部分，dp[i][j]=0
- 最长公共子数组以哪一项为末尾项都有可能，求出每个 dp[i][j]，找出最大值。


![2.png](http://ww1.sinaimg.cn/large/007daNw2ly1gpfh22j20lj31880lktd2.jpg)


#### 代码
- 时间复杂度 O(n * m)O(n∗m)。 空间复杂度 O(n * m)O(n∗m)。 
- 降维后空间复杂度 O(n)O(n)，如果没有空间复杂度的要求，降不降都行。

```go
func findLength(A []int, B []int) int {
	m, n := len(A), len(B)
	dp, res := make([][]int, m+1), 0
	for i := 0; i <= m; i++ {
		dp[i] = make([]int, n+1)
	}
	for i := 1; i <= m; i++ {
		for j := 1; j <= n; j++ {
			if A[i-1] == B[j-1] {
				dp[i][j] = dp[i-1][j-1] + 1
			}
			if res < dp[i][j] {
				res = dp[i][j]
			}
		}
	}
	return res
}
```


#### 降维优化
dp[i][j] 只依赖上一行上一列的对角线的值，所以我们从右上角开始计算。
一维数组 dp ， dp[j] 是以 A[i-1], B[j-1] 为末尾项的最长公共子数组的长度

![3.png](http://ww1.sinaimg.cn/large/007daNw2ly1gpfh7zq2paj31ff0l8dm8.jpg)

```go
func findLength(A []int, B []int) int {
	m, n := len(A), len(B)
	dp, res := make([]int, m+1), 0
	for i := 1; i <= m; i++ {
		for j := n; j >= 1; j-- {
			if A[i-1] == B[j-1] {
				dp[j] = dp[j-1] + 1
			} else {
				dp[j] = 0
			}
			if res < dp[j] {
				res = dp[j]
			}
		}
	}
	return res
}
```

### 方法三：动态规划



思路及算法
- 如果 A[i] == B[j]，
- 那么我们知道 A[i:] 与 B[j:] 的最长公共前缀为 A[i + 1:] 与 B[j + 1:] 的最长公共前缀的长度加一，
- 否则我们知道 A[i:] 与 B[j:] 的最长公共前缀为零。

这样我们就可以提出动态规划的解法：
令 dp[i][j] 表示 A[i:] 和 B[j:] 的最长公共前缀，那么答案即为所有 dp[i][j] 中的最大值。
- 如果 A[i] == B[j]，那么 dp[i][j] = dp[i + 1][j + 1] + 1，否则 dp[i][j] = 0。


考虑到这里 dp[i][j] 的值从 dp[i + 1][j + 1] 转移得到，所以我们需要倒过来，首先计算 dp[len(A) - 1][len(B) - 1]，最后计算 dp[0][0]。


```go
func findLength(A []int, B []int) int {
	dp, res := make([][]int, len(A)+1), 0
	for i := range dp {
		dp[i] = make([]int, len(B)+1)
	}
	for i := len(A) - 1; i >= 0; i-- {
		for j := len(B) - 1; j >= 0; j-- {
			if A[i] == B[j] {
				dp[i][j] = dp[i+1][j+1] + 1
			} else {
				dp[i][j] = 0
			}
			if res < dp[i][j] {
				res = dp[i][j]
			}
		}
	}
	return res
}
```

复杂度分析

- 时间复杂度： O(N×M)。

- 空间复杂度： O(N×M)。

N 表示数组 A 的长度，M 表示数组 B 的长度。

空间复杂度还可以再优化，利用滚动数组可以优化到 O(min(N,M))。




[144. 二叉树的前序遍历](https://leetcode-cn.com/problems/binary-tree-preorder-traversal/)

### 方法一：递归

```go
func preorderTraversal(root *TreeNode) (res []int) {
	var preorder func(*TreeNode)
	preorder = func(node *TreeNode) {
		if node != nil {
			res = append(res, node.Val)
			preorder(node.Left)
			preorder(node.Right)
		}
	}
	preorder(root)
	return
}
```

```go
func preorderTraversal(root *TreeNode) (res []int) {
	if root != nil {
		res = append(res, root.Val)
		tmp := preorderTraversal(root.Left)
		for _, t := range tmp {
			res = append(res, t)
		}
		tmp = preorderTraversal(root.Right)
		for _, t := range tmp {
			res = append(res, t)
		}
	}
	return
}
```

```go
var res []int

func preorderTraversal(root *TreeNode) []int {
	res = []int{}
	preorder(root)
	return res
}
func preorder(node *TreeNode) {
	if node != nil {
		res = append(res, node.Val)
		preorder(node.Left)
		preorder(node.Right)
	}
}
```

```go
func preorderTraversal(root *TreeNode) []int {
	res := []int{}
	preorder(root, &res)
	return res
}
func preorder(node *TreeNode, res *[]int) {
	if node != nil {
		*res = append(*res, node.Val)
		preorder(node.Left, res)
		preorder(node.Right, res)
	}
}
```

复杂度分析

- 时间复杂度：O(n)，其中 n 是二叉树的节点数。每一个节点恰好被遍历一次。
- 空间复杂度：O(n)，为递归过程中栈的开销，平均情况下为 O(logn)，最坏情况下树呈现链状，为 O(n)。

### 方法二：迭代


```go
func preorderTraversal(root *TreeNode) (res []int) {
	stack, node := []*TreeNode{}, root
	for node != nil || len(stack) > 0 {
		for node != nil {
			res = append(res, node.Val)
			stack = append(stack, node)
			node = node.Left
		}
		node = stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		node = node.Right
	}
	return
}
```

```go
func preorderTraversal(root *TreeNode) (res []int) {
	if root == nil {
		return []int{}
	}
	stack := []*TreeNode{root}
	for len(stack) > 0 {
		node := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		if node != nil {
			res = append(res, node.Val)
		}
		if node.Right != nil {
			stack = append(stack, node.Right)
		}
		if node.Left != nil {
			stack = append(stack, node.Left)
		}
	}
	return
}
```

复杂度分析

- 时间复杂度：O(n)，其中 n 是二叉树的节点数。每一个节点恰好被遍历一次。

- 空间复杂度：O(n)，为迭代过程中显式栈的开销，平均情况下为 O(logn)，最坏情况下树呈现链状，为 O(n)。




[169. 多数元素](https://leetcode-cn.com/problems/majority-element/)

### 方法五：Boyer-Moore 投票算法
思路

如果我们把众数记为 +1，把其他数记为 −1，将它们全部加起来，显然和大于 0，从结果本身我们可以看出众数比其他数多。

不同元素相互抵消，最后剩余就是众数

```go
func majorityElement(nums []int) int {
	res, count := 0, 0
	for _, num := range nums {
		if count == 0 {
			res = num
		}
		if res == num {
			count++
		} else {
			count--
		}
	}
	return res
}
```

```go
func majorityElement(nums []int) int {
	res, count := 0, 0
	for _, num := range nums {
		if count == 0 {
			res, count = num, 1
		} else {
			if res == num {
				count++
			} else {
				count--
			}
		}
	}
	return res
}
```

复杂度分析

- 时间复杂度：O(n)。Boyer-Moore 算法只对数组进行了一次遍历。

- 空间复杂度：O(1)。Boyer-Moore 算法只需要常数级别的额外空间。





[113. 路径总和 II](https://leetcode-cn.com/problems/path-sum-ii/)

### 方法一：深度优先搜索

思路及算法

我们可以采用深度优先搜索的方式，枚举每一条从根节点到叶子节点的路径。当我们遍历到叶子节点，且此时路径和恰为目标和时，我们就找到了一条满足条件的路径。

```go
func pathSum(root *TreeNode, targetSum int) [][]int {
	var res [][]int
	res = findPath(root, targetSum, res, []int(nil))
	return res
}
func findPath(node *TreeNode, sum int, res [][]int, stack []int) [][]int {
	if node == nil {
		return res
	}
	sum -= node.Val
	stack = append(stack, node.Val)
	if sum == 0 && node.Left == nil && node.Right == nil {
		res = append(res, append([]int(nil), stack...))
		stack = stack[:len(stack)-1]
	}
	res = findPath(node.Left, sum, res, stack)
	res = findPath(node.Right, sum, res, stack)
	return res
}
```


```go
func pathSum(root *TreeNode, targetSum int) (res [][]int) {
	stack := []int{} //path
	var dfs func(*TreeNode, int)
	dfs = func(node *TreeNode, sum int) {
		if node == nil {
			return
		}
		sum -= node.Val
		stack = append(stack, node.Val)
		defer func() { stack = stack[:len(stack)-1] }()
		if sum == 0 && node.Left == nil && node.Right == nil {
			res = append(res, append([]int(nil), stack...))
			return
		}
		dfs(node.Left, sum)
		dfs(node.Right, sum)
	}
	dfs(root, targetSum)
	return
}
```
复杂度分析

- 时间复杂度：O(N^2)，其中 N 是树的节点数。在最坏情况下，树的上半部分为链状，下半部分为完全二叉树，并且从根节点到每一个叶子节点的路径都符合题目要求。此时，路径的数目为 O(N)，并且每一条路径的节点个数也为 O(N)，因此要将这些路径全部添加进答案中，时间复杂度为 O(N^2)。

- 空间复杂度：O(N)，其中 N 是树的节点数。空间复杂度主要取决于栈空间的开销，栈中的元素个数不会超过树的节点数。



[234. 回文链表](https://leetcode-cn.com/problems/palindrome-linked-list/)

### 方法一：转成数组
遍历一遍，把值放入数组中，然后用双指针判断是否回文。

- 时间复杂度O(n)。
- 空间复杂度O(n)。


```go
/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
*/
func isPalindrome(head *ListNode) bool {
	nums := []int{}
	for head != nil {
		nums = append(nums, head.Val)
		head = head.Next
	}
	left, right := 0, len(nums)-1
	for left < right {
		if nums[left] != nums[right] {
			return false
		}
		left++
		right--
	}
	return true
}

```




### 方法二：快慢指针
快慢指针，起初都指向表头，快指针一次走两步，慢指针一次走一步，遍历结束时：

- 要么，slow 正好指向中间两个结点的后一个。
- 要么，slow 正好指向中间结点。
用 prev 保存 slow 的前一个结点，通过prev.next = null断成两个链表。

将后半段链表翻转，和前半段从头比对。空间复杂度降为O(1)。

![1.png](http://ww1.sinaimg.cn/large/007daNw2ly1gpecx13c3rj30z20fcjv7.jpg)

### 如何翻转单链表
可以这么思考：一次迭代中，有哪些指针需要变动：

每个结点的 next 指针要变动。
指向表头的 slow 指针要变动。
需要有指向新链表表头的 head2 指针，它也要变。

![2.png](http://ww1.sinaimg.cn/large/007daNw2ly1gpecxtza7vj31ie0ogn3i.jpg)

```go
/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
*/
func isPalindrome(head *ListNode) bool {
	if head == nil || head.Next == nil {
		return true
	}
	slow, fast := head, head
	prev := new(ListNode) // var prev *ListNode = nil
	for fast != nil && fast.Next != nil {
		prev = slow
		slow = slow.Next
		fast = fast.Next.Next
	}
	prev.Next = nil //断开
	//翻转后半部分链表
	head2 := new(ListNode)
	for slow != nil { 
		t := slow.Next
		slow.Next = head2
		head2, slow = slow, t
	}
	for head != nil && head2 != nil {
		if head.Val != head2.Val {
			return false
		}
		head = head.Next
		head2 = head2.Next
	}
	return true
}
```


[876. 链表的中间结点](https://leetcode-cn.com/problems/middle-of-the-linked-list/)

```go
/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */
func middleNode(head *ListNode) *ListNode {
    slow, fast := head, head 
    for fast != nil && fast.Next != nil {
        slow = slow.Next
        fast = fast.Next.Next
    }
    return slow
}
```


[155. 最小栈](https://leetcode-cn.com/problems/min-stack/)

```go
type MinStack struct {
	stack, minStack []int
}

/** initialize your data structure here. */
func Constructor() MinStack {
	return MinStack{
		stack:    []int{},
		minStack: []int{math.MaxInt64},
	}
}

func (this *MinStack) Push(val int) {
	this.stack = append(this.stack, val)
	top := this.minStack[len(this.minStack)-1]
	this.minStack = append(this.minStack, min(val, top))
}

func (this *MinStack) Pop() {
	this.stack = this.stack[:len(this.stack)-1]
	this.minStack = this.minStack[:len(this.minStack)-1]
}

func (this *MinStack) Top() int {
	return this.stack[len(this.stack)-1]
}

func (this *MinStack) GetMin() int {
	return this.minStack[len(this.minStack)-1]
}

func min(x, y int) int {
	if x < y {
		return x
	}
	return y
}
```

```go
// MinStack define
type MinStack struct {
	stack, min []int
	l          int
}

/** initialize your data structure here. */

// Constructor155 define
func Constructor() MinStack {
	return MinStack{make([]int, 0), make([]int, 0), 0}
}

// Push define
func (this *MinStack) Push(x int) {
	this.stack = append(this.stack, x)
	if this.l == 0 {
		this.min = append(this.min, x)
	} else {
		min := this.GetMin()
		if x < min {
			this.min = append(this.min, x)
		} else {
			this.min = append(this.min, min)
		}
	}
	this.l++
}

func (this *MinStack) Pop() {
	this.l--
	this.min = this.min[:this.l]
	this.stack = this.stack[:this.l]
}

func (this *MinStack) Top() int {
	return this.stack[this.l-1]
}

func (this *MinStack) GetMin() int {
	return this.min[this.l-1]
}
```



[56. 合并区间](https://leetcode-cn.com/problems/merge-intervals/)


### 方法一：排序

#### 思路

如果我们按照区间的左端点排序，那么在排完序的列表中，可以合并的区间一定是连续的。如下图所示，标记为蓝色、黄色和绿色的区间分别可以合并成一个大区间，它们在排完序的列表中是连续的：

![0.png](http://ww1.sinaimg.cn/large/007daNw2ly1gpfnnsrl94j30qo07a0t1.jpg)


#### 算法

我们用数组 merged 存储最终的答案。

首先，我们将列表中的区间按照左端点升序排序。然后我们将第一个区间加入 merged 数组中，并按顺序依次考虑之后的每个区间：

- 如果当前区间的左端点在数组 merged 中最后一个区间的右端点之后，那么它们不会重合，我们可以直接将这个区间加入数组 merged 的末尾；

- 否则，它们重合，我们需要用当前区间的右端点更新数组 merged 中最后一个区间的右端点，将其置为二者的较大值。


#### 思路
prev 初始为第一个区间，cur 表示当前的区间，res 表示结果数组

- 开启遍历，尝试合并 prev 和 cur，合并后更新到 prev 上
- 因为合并后的新区间还可能和后面的区间重合，继续尝试合并新的 cur，更新给 prev
- 直到不能合并 —— prev[1] < cur[0]，此时将 prev 区间推入 res 数组

#### 合并的策略
- 原则上要更新prev[0]和prev[1]，即左右端:
prev[0] = min(prev[0], cur[0])
prev[1] = max(prev[1], cur[1])
- 但如果先按区间的左端排升序，就能保证 prev[0] < cur[0]
- 所以合并只需这条：prev[1] = max(prev[1], cur[1])
#### 易错点
我们是先合并，遇到不重合再推入 prev。
当考察完最后一个区间，后面没区间了，遇不到不重合区间，最后的 prev 没推入 res。
要单独补上。


![1.png](http://ww1.sinaimg.cn/large/007daNw2ly1gpfnod6g2lj318c0ff760.jpg)

```go
func merge(intervals [][]int) [][]int {
	sort.Slice(intervals, func(i, j int) bool {
		return intervals[i][0] < intervals[j][0]
	})
	res := [][]int{}
	prev := intervals[0]
	for i := 1; i < len(intervals); i++ {
		curr := intervals[i]
		if prev[1] < curr[0] { //没有一点重合
			res = append(res, prev)
			prev = curr
		} else { //有重合
			prev[1] = max(prev[1], curr[1])
		}
	}
	res = append(res, prev)
	return res
}
func max(x, y int) int {
	if x > y {
		return x
	}
	return y
}
```

复杂度分析

- 时间复杂度：O(nlogn)，其中 nn 为区间的数量。除去排序的开销，我们只需要一次线性扫描，所以主要的时间开销是排序的 O(nlogn)。

- 空间复杂度：O(logn)，其中 nn 为区间的数量。这里计算的是存储答案之外，使用的额外空间。O(logn) 即为排序所需要的空间复杂度。


### sort.Slice()介绍

```go

import (
    "fmt"
)
func findRelativeRanks(nums []int){
    index := []int{}
    for i := 0; i < len(nums);i++ {
        index=append(index,i)
    }
    // index
    // {0,1,2,3,4}
    sort.Slice(index, func (i,j int)bool{
        return nums[i]<nums[j]
    })
    fmt.Println(index)
    // actually get [1 0 2 4 3], not [1, 4, 2, 3, 0] as expected
    // can't use sort.Slice between two slices
}
 
func main(){
    nums := []int{10,3,8,9,4}
    findRelativeRanks(nums)
```





[105. 从前序与中序遍历序列构造二叉树](https://leetcode-cn.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/)


```go
/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */
func buildTree(preorder []int, inorder []int) *TreeNode {
	if len(preorder) == 0 {
		return nil
	}
	root := &TreeNode{preorder[0], nil, nil}
	i := 0
	for ; i < len(inorder); i++ {
		if inorder[i] == preorder[0] { //找到根结点在 inorder 中的位置索引 i
			break
		}
	}
	root.Left = buildTree(preorder[1:len(inorder[:i])+1], inorder[:i])
	root.Right = buildTree(preorder[len(inorder[:i])+1:], inorder[i+1:])
	return root
}
```



[124. 二叉树中的最大路径和](https://leetcode-cn.com/problems/binary-tree-maximum-path-sum/)



```go
func maxPathSum(root *TreeNode) int {
	maxSum := math.MinInt32
	var dfs func(*TreeNode) int
	dfs = func(root *TreeNode) int {
		if root == nil {
			return 0
		}
		left := dfs(root.Left)
		right := dfs(root.Right)
		innerMaxSum := left + root.Val + right
		maxSum = max(maxSum, innerMaxSum)
		outputMaxSum := root.Val + max(left, right)
		return max(outputMaxSum, 0)
	}
	dfs(root)
	return maxSum
}
func max(x, y int) int {
	if x > y {
		return x
	}
	return y
}
```

```go
func maxPathSum(root *TreeNode) int {
	maxSum := math.MinInt32
	var dfs func(*TreeNode) int
	dfs = func(root *TreeNode) int {
		if root == nil {
			return 0
		}
		left := max(dfs(root.Left), 0)
		right := max(dfs(root.Right), 0)
		innerMaxSum := left + root.Val + right
		maxSum = max(maxSum, innerMaxSum)
		return root.Val + max(left, right) 
	}
	dfs(root)
	return maxSum
}
func max(x, y int) int {
	if x > y {
		return x
	}
	return y
}
```













