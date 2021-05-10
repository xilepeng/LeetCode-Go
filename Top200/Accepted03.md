

[718. 最长重复子数组](https://leetcode-cn.com/problems/maximum-length-of-repeated-subarray/)

[144. 二叉树的前序遍历](https://leetcode-cn.com/problems/binary-tree-preorder-traversal/)

[110. 平衡二叉树](https://leetcode-cn.com/problems/balanced-binary-tree/)

[56. 合并区间](https://leetcode-cn.com/problems/merge-intervals/)

[105. 从前序与中序遍历序列构造二叉树](https://leetcode-cn.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/)

[124. 二叉树中的最大路径和](https://leetcode-cn.com/problems/binary-tree-maximum-path-sum/)

[113. 路径总和 II](https://leetcode-cn.com/problems/path-sum-ii/)

[155. 最小栈](https://leetcode-cn.com/problems/min-stack/)

[1143. 最长公共子序列](https://leetcode-cn.com/problems/longest-common-subsequence/)

[234. 回文链表](https://leetcode-cn.com/problems/palindrome-linked-list/)

[876. 链表的中间结点](https://leetcode-cn.com/problems/middle-of-the-linked-list/) 补充

[470. 用 Rand7() 实现 Rand10()](https://leetcode-cn.com/problems/implement-rand10-using-rand7/)

[143. 重排链表](https://leetcode-cn.com/problems/reorder-list/)

[543. 二叉树的直径](https://leetcode-cn.com/problems/diameter-of-binary-tree/)

[169. 多数元素](https://leetcode-cn.com/problems/majority-element/)

[19. 删除链表的倒数第 N 个结点](https://leetcode-cn.com/problems/remove-nth-node-from-end-of-list/)

[4. 寻找两个正序数组的中位数](https://leetcode-cn.com/problems/median-of-two-sorted-arrays/) 	next

[101. 对称二叉树](https://leetcode-cn.com/problems/symmetric-tree/)

[31. 下一个排列](https://leetcode-cn.com/problems/next-permutation/)

[148. 排序链表](https://leetcode-cn.com/problems/sort-list/)

[83. 删除排序链表中的重复元素](https://leetcode-cn.com/problems/remove-duplicates-from-sorted-list/)



------



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




[110. 平衡二叉树](https://leetcode-cn.com/problems/balanced-binary-tree/)

### 方法一：自顶向下的递归 (前序遍历)

具体做法类似于二叉树的前序遍历，即对于当前遍历到的节点，首先计算左右子树的高度，如果左右子树的高度差是否不超过 11，再分别递归地遍历左右子节点，并判断左子树和右子树是否平衡。这是一个自顶向下的递归的过程。


```go
func isBalanced(root *TreeNode) bool {
	if root == nil {
		return true
	}
	leftHeight := depth(root.Left)
	rightHeight := depth(root.Right)
	return abs(rightHeight-leftHeight) <= 1 && isBalanced(root.Left) && isBalanced(root.Right)
}
func depth(root *TreeNode) int {
	if root == nil {
		return 0
	}
	return max(depth(root.Left), depth(root.Right)) + 1
}
func abs(x int) int {
	if x < 0 {
		return -1 * x
	}
	return x
}
func max(x, y int) int {
	if x > y {
		return x
	}
	return y
}
```

复杂度分析

- 时间复杂度：O(n^2)，其中 nn 是二叉树中的节点个数。
最坏情况下，二叉树是满二叉树，需要遍历二叉树中的所有节点，时间复杂度是 O(n)。
对于节点 p，如果它的高度是 d，则 height(p) 最多会被调用 d 次（即遍历到它的每一个祖先节点时）。对于平均的情况，一棵树的高度 hh 满足 O(h)=O(logn)，因为 d≤h，所以总时间复杂度为 O(nlogn)。对于最坏的情况，二叉树形成链式结构，高度为 O(n)，此时总时间复杂度为 O(n^2)。

- 空间复杂度：O(n)，其中 n 是二叉树中的节点个数。空间复杂度主要取决于递归调用的层数，递归调用的层数不会超过 n。


### 方法二：自底向上的递归 (后序遍历)

方法一由于是自顶向下递归，因此对于同一个节点，函数 height 会被重复调用，导致时间复杂度较高。如果使用自底向上的做法，则对于每个节点，函数 height 只会被调用一次。

自底向上递归的做法类似于后序遍历，对于当前遍历到的节点，先递归地判断其左右子树是否平衡，再判断以当前节点为根的子树是否平衡。如果一棵子树是平衡的，则返回其高度（高度一定是非负整数），否则返回 −1。如果存在一棵子树不平衡，则整个二叉树一定不平衡。

```go
func isBalanced(root *TreeNode) bool {
	return depth(root) >= 0
}
func depth(root *TreeNode) int {
	if root == nil {
		return 0
	}
	leftHeight := depth(root.Left)
	rightHeight := depth(root.Right)
	if leftHeight == -1 || rightHeight == -1 || abs(rightHeight-leftHeight) > 1 {
		return -1
	}
	return max(depth(root.Left), depth(root.Right)) + 1
}
func abs(x int) int {
	if x < 0 {
		return -1 * x
	}
	return x
}
func max(x, y int) int {
	if x > y {
		return x
	}
	return y
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






[1143. 最长公共子序列](https://leetcode-cn.com/problems/longest-common-subsequence/)

状态转移方程：

![截屏2021-04-15 21.12.08.png](http://ww1.sinaimg.cn/large/007daNw2ly1gpkqtiu1u5j30t404it9a.jpg)

最终计算得到 dp[m][n] 即为 text1 和 text2 的最长公共子序列的长度。

![](https://pic.leetcode-cn.com/1617411822-KhEKGw-image.png)

```go
	m, n := len(text1), len(text2)
	dp := make([][]int, m+1)
	for i := range dp {
		dp[i] = make([]int, n+1)
	}
	for i := 1; i <= m; i++ {
		for j := 1; j <= n; j++ {
			if text1[i-1] == text2[j-1] {
				dp[i][j] = dp[i-1][j-1] + 1
			} else {
				dp[i][j] = max(dp[i-1][j], dp[i][j-1])
			}
		}
	}
	return dp[m][n]
}
func max(x, y int) int {
	if x > y {
		return x
	}
	return y
}
```


```go
func longestCommonSubsequence(text1 string, text2 string) int {
	m, n := len(text1), len(text2)
	dp := make([][]int, m+1)
	for i := range dp {
		dp[i] = make([]int, n+1)
	}
	for i, c1 := range text1 {
		for j, c2 := range text2 {
			if c1 == c2 {
				dp[i+1][j+1] = dp[i][j] + 1
			} else {
				dp[i+1][j+1] = max(dp[i+1][j], dp[i][j+1])
			}
		}
	}
	return dp[m][n]
}
func max(x, y int) int {
	if x > y {
		return x
	}
	return y
}
```



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





[470. 用 Rand7() 实现 Rand10()](https://leetcode-cn.com/problems/implement-rand10-using-rand7/)

```go
func rand10() int {
	t := (rand7()-1)*7 + rand7() //1~49
	if t > 40 {
		return rand10()
	}
	return (t-1)%10 + 1
}
```

```go
func rand10() int {
	for {
		row, col := rand7(), rand7()
		idx := col + (row-1)*7
		if idx <= 40 {
			return 1 + (idx-1)%10
		}
	}
}
```





[143. 重排链表](https://leetcode-cn.com/problems/reorder-list/)

```go
func reorderList(head *ListNode) {
	if head == nil || head.Next == nil {
		return
	}

	// 寻找中间结点
	p1 := head
	p2 := head
	for p2.Next != nil && p2.Next.Next != nil {
		p1 = p1.Next
		p2 = p2.Next.Next
	}
	// 反转链表后半部分  1->2->3->4->5->6 to 1->2->3->6->5->4
	preMiddle := p1
	curr := p1.Next
	for curr.Next != nil {
		next := curr.Next
		curr.Next = next.Next
		next.Next = preMiddle.Next
		preMiddle.Next = next
	}

	// 重新拼接链表  1->2->3->6->5->4 to 1->6->2->5->3->4
	p1 = head
	p2 = preMiddle.Next
	for p1 != preMiddle {
		preMiddle.Next = p2.Next
		p2.Next = p1.Next
		p1.Next = p2
		p1 = p2.Next
		p2 = preMiddle.Next
	}
}
```





[543. 二叉树的直径](https://leetcode-cn.com/problems/diameter-of-binary-tree/)

```go
func diameterOfBinaryTree(root *TreeNode) int {
	res := 1
	var depth func(*TreeNode) int
	depth = func(node *TreeNode) int {
		if node == nil {
			return 0
		}
		left := depth(node.Left)
		right := depth(node.Right)
		res = max(res, left+right+1)
		return max(left, right) + 1
	}
	depth(root)
	return res - 1
}
func max(x, y int) int {
	if x > y {
		return x
	}
	return y
}
```





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





[19. 删除链表的倒数第 N 个结点](https://leetcode-cn.com/problems/remove-nth-node-from-end-of-list/)

### 方法一：双指针
思路与算法

使用两个指针 first 和 second 同时对链表进行遍历，并且 first 比 second 超前 nn 个节点。当 first 遍历到链表的末尾时，second 就恰好处于倒数第 nn 个节点。


![](https://assets.leetcode-cn.com/solution-static/19/p3.png)

```go
func removeNthFromEnd(head *ListNode, n int) *ListNode {
	dummy := &ListNode{0, head}
	first, second := head, dummy
	for i := 0; i < n; i++ {
		first = first.Next
	}
	for ; first != nil; first = first.Next {
		second = second.Next
	}
	second.Next = second.Next.Next
	return dummy.Next
}
```
复杂度分析

- 时间复杂度：O(L)，其中 L 是链表的长度。
- 空间复杂度：O(1)。








[4. 寻找两个正序数组的中位数](https://leetcode-cn.com/problems/median-of-two-sorted-arrays/)	next




[101. 对称二叉树](https://leetcode-cn.com/problems/symmetric-tree/)
### 方法一：递归
```go
func isSymmetric(root *TreeNode) bool {
	return check(root, root)
}
func check(p, q *TreeNode) bool {
	if p == nil && q == nil {
		return true
	}
	if p == nil || q == nil {
		return false
	}
	return p.Val == q.Val && check(p.Left, q.Right) && check(p.Right, q.Left)
}
```
### 方法二：迭代
```go
func isSymmetric(root *TreeNode) bool {
	q := []*TreeNode{root, root}
	for 0 < len(q) {
		l, r := q[0], q[1]
		q = q[2:]
		if l == nil && r == nil {
			continue
		}
		if l == nil || r == nil {
			return false
		}
		if l.Val != r.Val {
			return false
		}
		q = append(q, l.Left)
		q = append(q, r.Right)

		q = append(q, l.Right)
		q = append(q, r.Left)
	}
	return true
}
```




[31. 下一个排列](https://leetcode-cn.com/problems/next-permutation/)

### 思路

1. 我们需要将一个左边的「较小数」与一个右边的「较大数」交换，以能够让当前排列变大，从而得到下一个排列。

2. 同时我们要让这个「较小数」尽量靠右，而「较大数」尽可能小。当交换完成后，「较大数」右边的数需要按照升序重新排列。这样可以在保证新排列大于原来排列的情况下，使变大的幅度尽可能小。


- 从低位挑一个大一点的数，换掉前面的小一点的一个数，实现变大。
- 变大的幅度要尽量小。


像 [3,2,1] 递减的，没有下一个排列，因为大的已经尽量往前排了，没法更大。

像 [1,5,2,4,3,2] 这种，我们希望它稍微变大。

从低位挑一个大一点的数，换掉前面的小一点的一个数。

于是，从右往左，寻找第一个比右邻居小的数。（把它换到后面去）

找到 1 5 (2) 4 3 2 中间这个 2，让它和它身后的一个数交换，轻微变大。

还是从右往左，寻找第一个比这个 2 微大的数。15 (2) 4 (3) 2，交换，变成 15 (3) 4 (2) 2。

这并未结束，变大的幅度可以再小一点，仟位变大了，后三位可以再小一点。

后三位是递减的，翻转，变成[1,5,3,2,2,4]，即为所求。



```go
func nextPermutation(nums []int) {
	i := len(nums) - 2                   // 向左遍历，i从倒数第二开始是为了nums[i+1]要存在
	for i >= 0 && nums[i] >= nums[i+1] { // 寻找第一个小于右邻居的数
		i--
	}
	if i >= 0 { // 这个数在数组中存在，从它身后挑一个数，和它换
		j := len(nums) - 1                 // 从最后一项，向左遍历
		for j >= 0 && nums[j] <= nums[i] { // 寻找第一个大于 nums[i] 的数
			j--
		}
		nums[i], nums[j] = nums[j], nums[i] // 两数交换，实现变大
	}
	// 如果 i = -1，说明是递减排列，如 3 2 1，没有下一排列，直接翻转为最小排列：1 2 3
	l, r := i+1, len(nums)-1
	for l < r { // i 右边的数进行翻转，使得变大的幅度小一些
		nums[l], nums[r] = nums[r], nums[l]
		l++
		r--
	}
}
```




[148. 排序链表](https://leetcode-cn.com/problems/sort-list/)


## 方法一：自底向上归并排序

![2.png](http://ww1.sinaimg.cn/large/007daNw2ly1gph692tsj7j30z70jw75l.jpg)

- 时间：O(nlogn)
- 空间：O(1)

自顶向下递归形式的归并排序，由于递归需要使用系统栈，递归的最大深度是 logn，所以需要额外 O(logn)的空间。
所以我们需要使用自底向上非递归形式的归并排序算法。
基本思路是这样的，总共迭代 logn次：

第一次，将整个区间分成连续的若干段，每段长度是2：[a0,a1],[a2,a3],…[an−2,an−1]
， 然后将每一段内排好序，小数在前，大数在后；
第二次，将整个区间分成连续的若干段，每段长度是4：[a0,…,a3],[a4,…,a7],…[an−4,…,an−1]
，然后将每一段内排好序，这次排序可以利用之前的结果，相当于将左右两个有序的半区间合并，可以通过一次线性扫描来完成；
依此类推，直到每段小区间的长度大于等于 n 为止；
另外，当 n 不是2的整次幂时，每次迭代只有最后一个区间会比较特殊，长度会小一些，遍历到指针为空时需要提前结束。

- 时间复杂度分析：整个链表总共遍历 logn 次，每次遍历的复杂度是 O(n)，所以总时间复杂度是 O(nlogn)。
- 空间复杂度分析：整个算法没有递归，迭代时只会使用常数个额外变量，所以额外空间复杂度是 O(1)



```go
/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */
func sortList(head *ListNode) *ListNode {
	n := 0
	for p := head; p != nil; p = p.Next { n++ }
	dummy := &ListNode{Next: head}
	for i := 1; i < n; i *= 2 {
		cur := dummy
		for j := 1; j+i <= n; j += i * 2 {
			p:= cur.Next
            q := p
			for k := 0; k < i; k++ { q = q.Next }
			x, y := 0, 0
			for x < i && y < i && p != nil && q != nil {
				if p.Val <= q.Val {
					cur.Next = p
					cur = cur.Next
					p = p.Next
					x++
				} else {
					cur.Next = q
					cur = cur.Next
					q= q.Next
					y++
				}
			}
			for x < i && p != nil {
				cur.Next = p
				cur = cur.Next
				p = p.Next
				x++
			}
			for y < i && q != nil {
				cur.Next = q
				cur = cur.Next
				q = q.Next
				y++
			}
			cur.Next = q
		}
	}
	return dummy.Next
}
```


#### 解释

```go
/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */
func sortList(head *ListNode) *ListNode {
    //利用自底向上的归并思想，每次先归并好其中一小段，之后对两小段之间进行归并
	n := 0
	for p := head; p != nil; p = p.Next { n++ }
	dummy := &ListNode{Next: head}
    //每次归并段的长度，每次长度依次为1,2,4,8...n/2
    //小于n是因为等于n时说明所有元素均归并完毕，大于n时同理
	for i := 1; i < n; i *= 2 {
		cur := dummy
        //j代表每一段的开始，每次将两段有序段归并为一个大的有序段，故而每次+2i
        //必须保证每段中间序号是小于链表长度的，显然，如果大于表长，就没有元素可以归并了
		for j := 1; j+i <= n; j += i * 2 {
			p:= cur.Next //p表示第一段的起始点，q表示第二段的起始点，之后开始归并即可
            q := p
			for k := 0; k < i; k++ { q = q.Next }
            //x,y用于计数第一段和第二段归并的节点个数，由于当链表长度非2的整数倍时表长会小于i,
            //故而需要加上p != nil && q != nil的边界判断
			x, y := 0, 0 
			for x < i && y < i && p != nil && q != nil {
				if p.Val <= q.Val {//将小的一点插入cur链表中
					cur.Next = p
					cur = cur.Next
					p = p.Next
					x++
				} else {
					cur.Next = q
					cur = cur.Next
					q= q.Next
					y++
				}
			}
            //归并排序基本套路
			for x < i && p != nil {
				cur.Next = p
				cur = cur.Next
				p = p.Next
				x++
			}
			for y < i && q != nil {
				cur.Next = q
				cur = cur.Next
				q = q.Next
				y++
			}
			cur.Next = q //排好序的链表尾链接到下一链表的表头，循环完毕后q为下一链表表头
		}
	}
	return dummy.Next
}
```

#### C++ 代码
```C++
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    ListNode* sortList(ListNode* head) {
        int n = 0;
        for (auto p = head; p; p = p->next) n ++ ;

        auto dummy = new ListNode(-1);
        dummy->next = head;
        for (int i = 1; i < n; i *= 2) {
            auto cur = dummy;
            for (int j = 1; j + i <= n; j += i * 2) {
                auto p = cur->next, q = p;
                for (int k = 0; k < i; k ++ ) q = q->next;
                int x = 0, y = 0;
                while (x < i && y < i && p && q) {
                    if (p->val <= q->val) cur = cur->next = p, p = p->next, x ++ ;
                    else cur = cur->next = q, q = q->next, y ++ ;
                }
                while (x < i && p) cur = cur->next = p, p = p->next, x ++ ;
                while (y < i && q) cur = cur->next = q, q = q->next, y ++ ;
                cur->next = q;
            }
        }

        return dummy->next;
    }
};
```

## 方法二：自顶向下归并排序

思路
看到时间复杂度的要求，而且是链表，归并排序比较好做。
都知道归并排序要先归（二分），再并。两个有序的链表是比较容易合并的。
二分到不能再二分，即递归压栈压到链只有一个结点（有序），于是在递归出栈时进行合并。

合并两个有序的链表，合并后的结果返回给父调用，一层层向上，最后得出大问题的答案。

伪代码：
```go
func sortList (head) {
	对链表进行二分
	l = sortList(左链) // 已排序的左链
	r = sortList(右链) // 已排序的右链
	merged = mergeList(l, r) // 进行合并
	return merged		// 返回合并的结果给父调用
}
```

![3.png](http://ww1.sinaimg.cn/large/007daNw2ly1gph6854l3wj31i80luwj5.jpg)

#### 二分、merge 函数
- 二分，用快慢指针法，快指针走两步，慢指针走一步，快指针越界时，慢指针正好到达中点，只需记录慢指针的前一个指针，就能断成两链。
- merge 函数做的事是合并两个有序的左右链
    1. 设置虚拟头结点，用 prev 指针去“穿针引线”，prev 初始指向 dummy
    2. 每次都确定 prev.Next 的指向，并注意 l1，l2指针的推进，和 prev 指针的推进
    3. 最后返回 dummy.Next ，即合并后的链。


![4.png](http://ww1.sinaimg.cn/large/007daNw2ly1gph68ecpsoj31gc0iq43c.jpg)


```go
/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */
func sortList(head *ListNode) *ListNode {
	if head == nil || head.Next == nil {
		return head
	}
	slow, fast := head, head
	preSlow := new(ListNode)
	for fast != nil && fast.Next != nil {
		preSlow = slow
		slow = slow.Next
		fast = fast.Next.Next
	}
	preSlow.Next = nil
	left := sortList(head)
	right := sortList(slow)
	return mergeList(left, right)
}
func mergeList(l1, l2 *ListNode) *ListNode {
	if l1 == nil {
		return l2
	}
	if l2 == nil {
		return l1
	}
	if l1.Val < l2.Val {
		l1.Next = mergeList(l1.Next, l2)
		return l1
	} else {
		l2.Next = mergeList(l1, l2.Next)
		return l2
	}
}
```

```go
func sortList(head *ListNode) *ListNode {
	if head == nil || head.Next == nil { // 递归的出口，不用排序 直接返回
		return head
	}
	slow, fast := head, head // 快慢指针
	var preSlow *ListNode    // 保存slow的前一个结点
	for fast != nil && fast.Next != nil {
		preSlow = slow
		slow = slow.Next      // 慢指针走一步
		fast = fast.Next.Next // 快指针走两步
	}
	preSlow.Next = nil  // 断开，分成两链
	l := sortList(head) // 已排序的左链
	r := sortList(slow) // 已排序的右链
	return mergeList(l, r) // 合并已排序的左右链，一层层向上返回
}

func mergeList(l1, l2 *ListNode) *ListNode {
	dummy := &ListNode{Val: 0}   // 虚拟头结点
	prev := dummy                // 用prev去扫，先指向dummy
	for l1 != nil && l2 != nil { // l1 l2 都存在
		if l1.Val < l2.Val {   // l1值较小
			prev.Next = l1 // prev.Next指向l1
			l1 = l1.Next   // 考察l1的下一个结点
		} else {
			prev.Next = l2
			l2 = l2.Next
		}
		prev = prev.Next // prev.Next确定了，prev指针推进
	}
	if l1 != nil {    // l1存在，l2不存在，让prev.Next指向l1
		prev.Next = l1
	}
	if l2 != nil {
		prev.Next = l2
	}
	return dummy.Next // 真实头结点
}

```
#### 复杂度分析

- 时间复杂度：O(nlogn)，其中 n 是链表的长度。

- 空间复杂度：O(logn)，其中 n 是链表的长度。空间复杂度主要取决于递归调用的栈空间。



![1.png](http://ww1.sinaimg.cn/large/007daNw2ly1gph68oyr9nj30z70qb0tr.jpg)




[83. 删除排序链表中的重复元素](https://leetcode-cn.com/problems/remove-duplicates-from-sorted-list/)

```go
	/**
	 * Definition for singly-linked list.
	 * type ListNode struct {
	 *     Val int
	 *     Next *ListNode
	 * }
	 */
	func deleteDuplicates(head *ListNode) *ListNode {
		if head == nil {
			return nil
		}
		curr := head
		for curr.Next != nil {
			if curr.Val == curr.Next.Val {
				curr.Next = curr.Next.Next
			} else {
				curr = curr.Next
			}
		}
		return head
	}
```

