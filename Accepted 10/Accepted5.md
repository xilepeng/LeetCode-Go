[718. 最长重复子数组](https://leetcode-cn.com/problems/maximum-length-of-repeated-subarray/)

[70. 爬楼梯](https://leetcode-cn.com/problems/climbing-stairs/)

[113. 路径总和 II](https://leetcode-cn.com/problems/path-sum-ii/)

[8. 字符串转换整数 (atoi)](https://leetcode-cn.com/problems/string-to-integer-atoi/)

[234. 回文链表](https://leetcode-cn.com/problems/palindrome-linked-list/)

[155. 最小栈](https://leetcode-cn.com/problems/min-stack/)

[543. 二叉树的直径](https://leetcode-cn.com/problems/diameter-of-binary-tree/)

[105. 从前序与中序遍历序列构造二叉树](https://leetcode-cn.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/)

[124. 二叉树中的最大路径和](https://leetcode-cn.com/problems/binary-tree-maximum-path-sum/)

[4. 寻找两个正序数组的中位数](https://leetcode-cn.com/problems/median-of-two-sorted-arrays/)


------


[70. 爬楼梯](https://leetcode-cn.com/problems/climbing-stairs/)


### 方法一：滚动数组 (斐波那契数列)

![截屏2021-04-09 11.50.02.png](http://ww1.sinaimg.cn/large/007daNw2ly1gpdcsbx511j31940gu756.jpg)

思路和算法

我们用 f(x) 表示爬到第 x 级台阶的方案数，考虑最后一步可能跨了一级台阶，也可能跨了两级台阶，所以我们可以列出如下式子：

f(x) = f(x - 1) + f(x - 2)

f(x) 只和 f(x−1) 与 f(x−2) 有关，所以我们可以用「滚动数组思想」把空间复杂度优化成 O(1)。


```go
func climbStairs(n int) int {
	p, q, r := 0, 0, 1
	for i := 1; i <= n; i++ {
		p = q
		q = r
		r = p + q
	}
	return r
}
```
复杂度分析

- 时间复杂度：循环执行 n 次，每次花费常数的时间代价，故渐进时间复杂度为 O(n)。
- 空间复杂度：这里只用了常数个变量作为辅助空间，故渐进空间复杂度为 O(1)。



### 方法二：动态规划

解题思路 
- 简单的 DP，经典的爬楼梯问题。一个楼梯可以由 n-1 和 n-2 的楼梯爬上来。
- 这一题求解的值就是斐波那契数列。

```go
func climbStairs(n int) int {
	dp := make([]int, n+1)
	dp[0], dp[1] = 1, 1
	for i := 2; i <= n; i++ {
		dp[i] = dp[i-1] + dp[i-2]
	}
	return dp[n]
}
```

复杂度分析

- 时间复杂度：循环执行 n 次，每次花费常数的时间代价，故渐进时间复杂度为 O(n)。
- 空间复杂度： O(n)。

### 解题思路 
#### 假设 n = 5，有 5 级楼梯要爬
- 题意说，每次有2种选择：爬1级，或爬2级。
	如果爬1级，则剩下4级要爬。
	如果爬2级，则剩下3级要爬。
- 这拆分出了2个问题：
	爬4级楼梯有几种方式？
	爬3级楼梯有几种方式？
- 于是，爬 5 级楼梯的方式数 = 爬 4 级楼梯的方式数 + 爬 3 级楼梯的方式数。
#### 画出递归树
- 用「剩下要爬的楼梯数」描述一个节点。
![1.png](http://ww1.sinaimg.cn/large/007daNw2ly1gpdsdqld5ij317207674g.jpg)

- 子问题又会面临 2 个选择，不断分支，直到位于递归树底部的 base case：

	楼梯数为 0 时，只有 1 种选择：什么都不做。
	楼梯数为 1 时，只有1种选择：爬1级。

![2.png](http://ww1.sinaimg.cn/large/007daNw2ly1gpdsewfog7j30y60ctaam.jpg)

- 当递归遍历到 base case，解是已知的，开始返回，如下图，子问题的结果不断向上返回，得到父问题的解。

![4.png](http://ww1.sinaimg.cn/large/007daNw2ly1gpdsfh45y0j314h0hkdhr.jpg)

- 调用栈的深度是楼梯数 n，空间复杂度是O(n)O(n)，时间复杂度最坏是O(2^n)，所有节点都遍历到。

### 存在重复的子问题
- 如下图，黄色阴影部分，蓝色阴影部分，是相同的子树，即相同的子问题。

![5.png](http://ww1.sinaimg.cn/large/007daNw2ly1gpdsh8p300j30zr0i7jtk.jpg)

- 子问题的计算结果可以存储在哈希表中，或者数组，下次遇到就不用再进入相同的递归。

- 去除重复的计算后的子树如下，时间复杂度降到了O(n)，空间复杂度O(n)。

![6.png](http://ww1.sinaimg.cn/large/007daNw2ly1gpdsiibb52j30sp0fpabc.jpg)

## 动态规划，自底而上思考
- 我们发现，爬 i 层楼梯的方式数 = 爬 i-2 层楼梯的方式数 + 爬 i-1 层楼梯的方式数。
- 我们有两个 base case，结合上面的递推式，就能递推出爬 i 层楼梯的方式数。
- 用一个数组 dp 存放中间子问题的结果。
- dp[i]：爬 i 级楼梯的方式数。从dp[0]、dp[1]出发，顺序计算，直到算出 dp[i]，就像填表格。

![7.png](http://ww1.sinaimg.cn/large/007daNw2ly1gpdskk5ycoj31c00kx781.jpg)

```go
func climbStairs(n int) int {
	dp := make([]int, n+1)
	dp[0] = 1
	dp[1] = 1
	for i := 2; i < len(dp); i++ {
		dp[i] = dp[i-2] + dp[i-1]
	}
	return dp[n]
}

```

## 压缩空间，优化
dp[i] 只与过去的两项：dp[i-1] 和 dp[i-2] 有关，没有必要存下所有计算过的 dp 项。用两个变量去存这两个过去的状态就好。

```go

func climbStairs(n int) int {
	prev := 1
	cur := 1
	for i := 2; i < n+1; i++ {
		temp := cur
		cur = prev + cur
		prev = temp
	}
	return cur
}

```

### 可以用动态规划的问题都能用递归
- 从子问题入手，解决原问题，分两种做法：自顶向下和自底向上
- 前者对应递归，借助函数调用自己，是程序解决问题的方式，它不会记忆解
- 后者对应动态规划，利用迭代将子问题的解存在数组里，从数组0位开始顺序往后计算
- 递归的缺点在于包含重复的子问题（没有加记忆化的情况下），动态规划的效率更高

### DP也有局限性
- DP 相比于 递归，有时候不太好理解，或者边界情况比较难确定
- 而且必须是一步步邻接的，连续地计算
- 加入了记忆化的递归，就灵活很多，它在递归基础上稍作修改，往往更好理解，也少了局限性，不好用DP时一定能用它
- 比如有时候要求出达到某个结果的路径，递归（DFS）回溯出路径，显然更有优势





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


[8. 字符串转换整数 (atoi)](https://leetcode-cn.com/problems/string-to-integer-atoi/)

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

# Challenge Accepted 10 	Done 2021.4.10




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
[4. 寻找两个正序数组的中位数](https://leetcode-cn.com/problems/median-of-two-sorted-arrays/)


