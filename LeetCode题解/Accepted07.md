

1. [224. 基本计算器](#224-基本计算器)
2. [662. 二叉树最大宽度](#662-二叉树最大宽度)
3. [349. 两个数组的交集](#349-两个数组的交集)
4. [230. 二叉搜索树中第K小的元素](#230-二叉搜索树中第k小的元素)
5. [297. 二叉树的序列化与反序列化](#297-二叉树的序列化与反序列化)
6. [221. 最大正方形](#221-最大正方形)
7. [79. 单词搜索](#79-单词搜索)
8. [9. 回文数](#9-回文数)
9. [剑指 Offer 10- II. 青蛙跳台阶问题](#剑指-offer-10--ii-青蛙跳台阶问题)
10. [剑指 Offer 10- I. 斐波那契数列](#剑指-offer-10--i-斐波那契数列)
11. [70. 爬楼梯](#70-爬楼梯)
12. [剑指 Offer 62. 圆圈中最后剩下的数字](#剑指-offer-62-圆圈中最后剩下的数字)
13. [剑指 Offer 27. 二叉树的镜像](#剑指-offer-27-二叉树的镜像)
14. [226. 翻转二叉树](#226-翻转二叉树)
	1. [方法一：dfs 递归](#方法一dfs-递归)
	2. [方法一：dfs 递归](#方法一dfs-递归-1)
	3. [方法二: BFS](#方法二-bfs)
15. [补充题23. 检测循环依赖](#补充题23-检测循环依赖)
	1. [拓扑排序算法过程：](#拓扑排序算法过程)
16. [207. 课程表](#207-课程表)
	1. [方法一: 广度优先搜索](#方法一-广度优先搜索)
	2. [方法二：深度优先搜索](#方法二深度优先搜索)
17. [210. 课程表 II](#210-课程表-ii)
18. [739. 每日温度](#739-每日温度)
19. [26. 删除有序数组中的重复项](#26-删除有序数组中的重复项)
20. [287. 寻找重复数](#287-寻找重复数)
	1. [方法一：快慢指针](#方法一快慢指针)
	2. [方法二：二分查找](#方法二二分查找)
	3. [方法三：排序](#方法三排序)
	4. [方法四：哈希](#方法四哈希)
21. [11. 盛最多水的容器](#11-盛最多水的容器)
22. [560. 和为K的子数组](#560-和为k的子数组)
23. [443. 压缩字符串](#443-压缩字符串)
24. [50. Pow(x, n)](#50-powx-n)
25. [补充题2. 圆环回原点问题](#补充题2-圆环回原点问题)



<!-- [224. 基本计算器](https://leetcode-cn.com/problems/basic-calculator/)

[662. 二叉树最大宽度](https://leetcode-cn.com/problems/maximum-width-of-binary-tree/)

[349. 两个数组的交集](https://leetcode-cn.com/problems/intersection-of-two-arrays/)

[230. 二叉搜索树中第K小的元素](https://leetcode-cn.com/problems/kth-smallest-element-in-a-bst/)

[297. 二叉树的序列化与反序列化](https://leetcode-cn.com/problems/serialize-and-deserialize-binary-tree/)

[221. 最大正方形](https://leetcode-cn.com/problems/maximal-square/)

[79. 单词搜索](https://leetcode-cn.com/problems/word-search/)

[9. 回文数](https://leetcode-cn.com/problems/palindrome-number/)

[剑指 Offer 10- II. 青蛙跳台阶问题](https://leetcode-cn.com/problems/qing-wa-tiao-tai-jie-wen-ti-lcof/)

[剑指 Offer 62. 圆圈中最后剩下的数字](https://leetcode-cn.com/problems/yuan-quan-zhong-zui-hou-sheng-xia-de-shu-zi-lcof/)

[剑指 Offer 27. 二叉树的镜像](https://leetcode-cn.com/problems/er-cha-shu-de-jing-xiang-lcof/)

[补充题23. 检测循环依赖](https://mp.weixin.qq.com/s/q6AhBt6MX2RL_HNZc8cYKQ)

[739. 每日温度](https://leetcode-cn.com/problems/daily-temperatures/)

[26. 删除有序数组中的重复项](https://leetcode-cn.com/problems/remove-duplicates-from-sorted-array/)

[287. 寻找重复数](https://leetcode-cn.com/problems/find-the-duplicate-number/)

[11. 盛最多水的容器](https://leetcode-cn.com/problems/container-with-most-water/)

[560. 和为K的子数组](https://leetcode-cn.com/problems/subarray-sum-equals-k/)

[443. 压缩字符串](https://leetcode-cn.com/problems/string-compression/)

[50. Pow(x, n)](https://leetcode-cn.com/problems/powx-n/)

[补充题2. 圆环回原点问题](https://mp.weixin.qq.com/s/VnGFEWHeD3nh1n9JSDkVUg) -->


------



## [224. 基本计算器](https://leetcode-cn.com/problems/basic-calculator/)

```go
func calculate(s string) int {
	stack, res, num, sign := []int{}, 0, 0, 1
	for i := 0; i < len(s); i++ {
		if s[i] >= '0' && s[i] <= '9' {
			num = num*10 + int(s[i]-'0')
		} else if s[i] == '+' {
			res += sign * num
			num = 0
			sign = 1
		} else if s[i] == '-' {
			res += sign * num
			num = 0
			sign = -1
		} else if s[i] == '(' {
			stack = append(stack, res) //将前一个结果和符号压入栈中
			stack = append(stack, sign)
			res = 0 //将结果设置为0，只需在括号内计算新结果
			sign = 1
		} else if s[i] == ')' {
			res += sign * num
			num = 0
			res *= stack[len(stack)-1]
			res += stack[len(stack)-2]
			stack = stack[:len(stack)-2]
		}
	}
	if num != 0 {
		res += sign * num
	}
	return res
}

```

## [662. 二叉树最大宽度](https://leetcode-cn.com/problems/maximum-width-of-binary-tree/)

```go
/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */
func widthOfBinaryTree(root *TreeNode) int {
	res, left := 0, []int{1}
	var dfs func(root *TreeNode, level, pos int)
	dfs = func(root *TreeNode, level, pos int) {
		if root != nil {
			if level == len(left) {
				left = append(left, pos)
			}
			res = max(res, pos-left[level]+1) //计算每层的最大间距
			dfs(root.Left, level+1, pos*2)
			dfs(root.Right, level+1, pos*2+1)
		}
	}
	dfs(root, 0, 1)
	return res
}
func max(x, y int) int {
	if x > y {
		return x
	}
	return y
}
```

## [349. 两个数组的交集](https://leetcode-cn.com/problems/intersection-of-two-arrays/)

```go
func intersection(nums1 []int, nums2 []int) []int {
	m := map[int]bool{}
	res := []int{}
	for _, num1 := range nums1 {
		m[num1] = true
	}
	for _, num2 := range nums2 {
		if m[num2] {
			m[num2] = false
			res = append(res, num2)
		}
	}
	return res
}
```

## [230. 二叉搜索树中第K小的元素](https://leetcode-cn.com/problems/kth-smallest-element-in-a-bst/)

**方法一：迭代**


```go
/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */

func kthSmallest(root *TreeNode, k int) int {
	stack := []*TreeNode{}
	for root != nil || len(stack) > 0 {
		if root != nil {
			stack = append(stack, root)
			root = root.Left
		} else {
			node := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			k--
			if k == 0 {
				return node.Val
			}
			root = node.Right
		}
	}
	return 0
}
```
复杂度分析

- 时间复杂度：O(H+k)，其中 H 指的是树的高度，由于我们开始遍历之前，要先向下达到叶，当树是一个平衡树时：复杂度为 O(logN+k)。当树是一个不平衡树时：复杂度为 O(N+k)，此时所有的节点都在左子树。

- 空间复杂度：O(H+k)。当树是一个平衡树时：O(logN+k)。当树是一个非平衡树时：O(N+k)



**方法二：递归**

```go
/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */
func kthSmallest(root *TreeNode, k int) int {
	res := 0
	var inorder func(root *TreeNode)

	inorder = func(root *TreeNode) {
		if root != nil {
			inorder(root.Left)
			k--
			if k == 0 {
				res = root.Val
				return
			}
			inorder(root.Right)
		}
	}
	inorder(root)
	return res
}
```


```go
/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */
func kthSmallest(root *TreeNode, k int) int {
	sorted := []int{}
	var inorder func(root *TreeNode)

	inorder = func(root *TreeNode) {
		if root != nil {
			inorder(root.Left)
			sorted = append(sorted, root.Val)
			inorder(root.Right)
		}
	}
	inorder(root)
	return sorted[k-1]
}
```
复杂度分析

- 时间复杂度：O(N)，遍历了整个树。
- 空间复杂度：O(N)，用了一个数组存储中序序列。



## [297. 二叉树的序列化与反序列化](https://leetcode-cn.com/problems/serialize-and-deserialize-binary-tree/)

```go
/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */

type Codec struct {
}

func Constructor() Codec {
	return Codec{}
}

// Serializes a tree to a single string.
func (this *Codec) serialize(root *TreeNode) string {
	if root == nil {
		return "X,"
	}
	left := this.serialize(root.Left)
	right := this.serialize(root.Right)

	return strconv.Itoa(root.Val) + "," + left + right
}

// Deserializes your encoded data to tree.
func (this *Codec) deserialize(data string) *TreeNode {
	nodes := strings.Split(data, ",")
	return this.buildTree(&nodes)
}

func (this *Codec) buildTree(nodes *[]string) *TreeNode {
	nodeVal := "X"
	if len(*nodes) > 0 {
		nodeVal = (*nodes)[0]
		*nodes = (*nodes)[1:]
	}
	if nodeVal == "X" {
		return nil
	} else {
		val, _ := strconv.Atoi(nodeVal)
		node := &TreeNode{Val: val}
		node.Left = this.buildTree(nodes)
		node.Right = this.buildTree(nodes)
		return node
	}
}

/**
 * Your Codec object will be instantiated and called as such:
 * ser := Constructor();
 * deser := Constructor();
 * data := ser.serialize(root);
 * ans := deser.deserialize(data);
 */
```

## [221. 最大正方形](https://leetcode-cn.com/problems/maximal-square/)

```go
func maximalSquare(matrix [][]byte) int {
	if len(matrix) == 0 || len(matrix[0]) == 0 {
		return 0
	}
	m, n, max := len(matrix), len(matrix[0]), 0
	dp := make([][]int, m+1)
	for i := range dp {
		dp[i] = make([]int, n+1)
	}
	for i := 1; i <= m; i++ {
		for j := 1; j <= n; j++ {
			if matrix[i-1][j-1] == '1' {
				dp[i][j] = min(dp[i-1][j-1], min(dp[i-1][j], dp[i][j-1])) + 1
			}
			if dp[i][j] > max {
				max = dp[i][j]
			}
		}
	}
	return max * max
}
func min(x, y int) int {
	if x < y {
		return x
	}
	return y
}
```

## [79. 单词搜索](https://leetcode-cn.com/problems/word-search/)

```go
func exist(board [][]byte, word string) bool {
	var dfs func(int, int, int) bool

	dfs = func(y, x, i int) bool {
		if i == len(word) {
			return true
		}
		if y < 0 || x < 0 || y == len(board) || x == len(board[y]) {
			return false
		}
		if board[y][x] != word[i] {
			return false
		}
		board[y][x] ^= 255
		exist := dfs(y, x+1, i+1) || dfs(y, x-1, i+1) || dfs(y+1, x, i+1) || dfs(y-1, x, i+1)
		board[y][x] ^= 255
		return exist
	}

	for y := 0; y < len(board); y++ {
		for x := 0; x < len(board[y]); x++ {
			if dfs(y, x, 0) {
				return true
			}
		}
	}
	return false
}
```

```go
func exist(board [][]byte, word string) bool {
	for y := 0; y < len(board); y++ {
		for x := 0; x < len(board[y]); x++ {
			if dfs(board, y, x, word, 0) {
				return true
			}
		}
	}
	return false
}
func dfs(board [][]byte, y int, x int, word string, i int) bool {
	if i == len(word) {
		return true
	}
	if y < 0 || x < 0 || y == len(board) || x == len(board[y]) {
		return false
	}
	if board[y][x] != word[i] {
		return false
	}
	board[y][x] ^= 255
	exist := dfs(board, y, x+1, word, i+1) || dfs(board, y, x-1, word, i+1) ||
		dfs(board, y+1, x, word, i+1) || dfs(board, y-1, x, word, i+1)
	board[y][x] ^= 255
	return exist
}
```


## [9. 回文数](https://leetcode-cn.com/problems/palindrome-number/)



```go
func isPalindrome(x int) bool {
	if x < 0 || (x%10 == 0 && x != 0) { //第1位不是0，最后一位是0
		return false
	}
	rev := 0
	for x > rev {
		rev = rev*10 + x%10
		x /= 10
	}
	return x == rev || x == rev/10 //奇数去除处于中位的数字
}

```

复杂度分析

时间复杂度：O(logn)，对于每次迭代，我们会将输入除以 10，因此时间复杂度为 O(logn)。
空间复杂度：O(1)。我们只需要常数空间存放若干变量。



```go
func isPalindrome(x int) bool {
	if x < 0 {
		return false
	}
	s := strconv.Itoa(x)
	left, right := 0, len(s)-1
	for left < right {
		if s[left] != s[right] {
			return false
		}
		left++
		right--
	}
	return true
}
```


## [剑指 Offer 10- II. 青蛙跳台阶问题](https://leetcode-cn.com/problems/qing-wa-tiao-tai-jie-wen-ti-lcof/)

```go
func numWays(n int) int {
    prev, curr := 1, 1
    for i := 2; i <= n; i++ {
        next := (prev+curr)%1000000007
        prev = curr
        curr = next
    }
    return curr
}
```


```go
func numWays(n int) int {
	a, b := 1, 1
	for i := 0; i < n; i++ {
		a, b = b, (a+b)%1000000007
	}
	return a
}
```

## [剑指 Offer 10- I. 斐波那契数列](https://leetcode-cn.com/problems/fei-bo-na-qi-shu-lie-lcof/)

```go
func fib(n int) int {
    if n == 0 || n == 1 {
        return n 
    }
    prev, curr := 0, 1
    for i := 2; i <= n; i++ {
        next := (prev + curr)%1000000007
        prev = curr
        curr = next
    }
    return curr
}
```

```go
func fib(n int) int {
    a, b := 0, 1
    for i := 0; i < n; i++ {
        a, b = b, (a+b)%1000000007
    }
    return a
}
```




## [70. 爬楼梯](https://leetcode-cn.com/problems/climbing-stairs/)

```go
func climbStairs(n int) int {
	prev, curr := 1, 1
	for i := 2; i <= n; i++ {
		next := prev + curr
		prev = curr
		curr = next
	}
	return curr
}
```

```go
func climbStairs(n int) int {
	a, b := 1, 1
	for i := 0; i < n; i++ {
		a, b = b, a+b
	}
	return a
}
```


## [剑指 Offer 62. 圆圈中最后剩下的数字](https://leetcode-cn.com/problems/yuan-quan-zhong-zui-hou-sheng-xia-de-shu-zi-lcof/)


```go
func lastRemaining(n int, m int) int {
    f := 0
    for i := 2; i <= n; i++ {
        f = (f+m)%i
    }
    return f
}
```


## [剑指 Offer 27. 二叉树的镜像](https://leetcode-cn.com/problems/er-cha-shu-de-jing-xiang-lcof/)

![](images/Howell.png)


## [226. 翻转二叉树](https://leetcode-cn.com/problems/invert-binary-tree/)

### 方法一：dfs 递归
它的左右子树要交换，并且左右子树内部的所有子树，都要进行左右子树的交换。

![undefined](http://ww1.sinaimg.cn/large/007daNw2ly1gqxsqrjnjmj61er0fw0w102.jpg)

每个子树的根节点都说：先交换我的左右子树吧。那么递归就会先压栈压到底。然后才做交换。
即，位于底部的、左右孩子都是 null 的子树，先被翻转。
随着递归向上返回，子树一个个被翻转……整棵树翻转好了。
问题是在递归出栈时解决的。


```go
/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */
func invertTree(root *TreeNode) *TreeNode {
	if root == nil {
		return nil
	}
	invertTree(root.Left)                         //递归左子树 (压栈压到底部)
	invertTree(root.Right)                        //递归右子树
	root.Left, root.Right = root.Right, root.Left //自底向上交换左右子树
	return root
}
```

### 方法一：dfs 递归

![undefined](http://ww1.sinaimg.cn/large/007daNw2ly1gqxszzqq13j31fu0gh77e.jpg)

思路变了：先 “做事”——先交换左右子树，它们内部的子树还没翻转——丢给递归去做。
把交换的操作，放在递归子树之前。
问题是在递归压栈前被解决的。

```go
/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */
func invertTree(root *TreeNode) *TreeNode {
	if root == nil {
		return nil
	}
	root.Left, root.Right = root.Right, root.Left //交换左右子树
	invertTree(root.Left)                         //递归左子树 (压栈压到底部)
	invertTree(root.Right)                        //递归右子树
	return root
}
```


### 方法二: BFS 
用层序遍历的方式去遍历二叉树。

根节点先入列，然后出列，出列就 “做事”，交换它的左右子节点（左右子树）。
并让左右子节点入列，往后，这些子节点出列，也被翻转。
直到队列为空，就遍历完所有的节点，翻转了所有子树。

解决问题的代码放在节点出列时。


```go
/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */
func invertTree(root *TreeNode) *TreeNode {
	if root == nil {
		return nil
	}
	queue := []*TreeNode{root}
	for len(queue) != 0 {
		node := queue[0]
		queue = queue[1:]
		node.Left, node.Right = node.Right, node.Left //交换左右子树
		if node.Left != nil {
			queue = append(queue, node.Left)
		}
		if node.Right != nil {
			queue = append(queue, node.Right)
		}
	}
	return root
}
```












## [补充题23. 检测循环依赖](https://mp.weixin.qq.com/s/q6AhBt6MX2RL_HNZc8cYKQ)


### 拓扑排序算法过程：

1. 选择图中一个入度为0的点，记录下来
2. 在图中删除该点和所有以它为起点的边
3. 重复1和2，直到图为空或没有入度为0的点。


```go
/*
 * @lc app=leetcode.cn id=210 lang=golang
 *
 * [补充题]：检测循环依赖
 */

// @lc code=start

func findOrder(n int, prerequisites [][]int) []int {
	g := make([][]int, n)       //邻接表存储图结构
	in_degree := make([]int, n) //每个点的入度
	res := []int{}
	// 获取每个点的入度和邻接
	for _, pre := range prerequisites {
		g[pre[0]] = append(g[pre[0]], pre[1]) // pre[0] -> pre[1]
		in_degree[pre[1]]++
	}

	// for _, pre := range prerequisites {
	// 	g[pre[1]] = append(g[pre[1]], pre[0]) // pre[0] <- pre[1]
	// 	in_degree[pre[0]]++
	// }

	q := []int{}
	for i := 0; i < n; i++ {
		if in_degree[i] == 0 { // 将所有入度为 0 的节点放入队列中
			q = append(q, i)
		}
	}

	for len(q) != 0 {
		u := q[0]
		q = q[1:]
		res = append(res, u)
		for _, v := range g[u] {
			in_degree[v]-- //在图中删除该点和所有以它为起点的边
			if in_degree[v] == 0 {
				q = append(q, v)
			}
		}
	}

	if len(res) != n { //有循环依赖
		return []int{}
	} 
	return res

}
```



## [207. 课程表](https://leetcode-cn.com/problems/course-schedule/)


### 方法一: 广度优先搜索

```go
func canFinish(numCourses int, prerequisites [][]int) bool {
	T := make([][]int, numCourses)       // 存储有向图
	in_degree := make([]int, numCourses) // 存储每个节点的入度
	res := make([]int, 0, numCourses)    // 存储答案

	for _, v := range prerequisites { // 获取每门课程的入度和邻接
		T[v[1]] = append(T[v[1]], v[0]) // v[1] -> v[0] 学 v[0] 前要学 v[1]
		in_degree[v[0]]++
	}

	q := []int{} // 将所有入度为 0 的节点放入队列中
	for i := 0; i < numCourses; i++ {
		if in_degree[i] == 0 {
			q = append(q, i)
		}
	}

	for len(q) > 0 { // BFS Topological Sort 拓扑排序
		u := q[0] // 从队首取出一个节点
		q = q[1:]
		res = append(res, u) // 放入答案中
		for _, v := range T[u] {
			in_degree[v]--
			if in_degree[v] == 0 { // 如果相邻节点 v 的入度为 0，可以选 v 对应的课程了
				q = append(q, v)
			}
		}
	}
	return len(res) == numCourses // 无环
}
```

### 方法二：深度优先搜索

```go
func canFinish(numCourses int, prerequisites [][]int) bool {
	T := make([][]int, numCourses)       // 存储有向图
	in_degree := make([]int, numCourses) // 存储每个节点的入度
	visited := make([]int, numCourses)
	var findCircle func(int) bool

	findCircle = func(node int) bool { // DFS Topological Sort 拓扑排序
		if visited[node] == 1 {
			return true //有环
		}
		if visited[node] == 2 {
			return false
		}
		visited[node] = 1
		for _, next := range T[node] {
			if findCircle(next) {
				return true
			}
		}
		visited[node] = 2 //已访问
		return false
	}

	for _, v := range prerequisites { // 获取每门课程的入度和邻接
		T[v[1]] = append(T[v[1]], v[0]) // v[1] -> v[0] 学 v[0] 前要学 v[1]
		in_degree[v[0]]++
	}
	for i := 0; i < numCourses; i++ {
		if findCircle(i) {
			return false
		}
	}
	return true
}
```





## [210. 课程表 II](https://leetcode-cn.com/problems/course-schedule-ii/)

```go
func findOrder(numCourses int, prerequisites [][]int) []int {
	graph := make([][]int, numCourses)
	in_degree := make([]int, numCourses)
	res := []int{}
	for _, pre := range prerequisites {
		graph[pre[1]] = append(graph[pre[1]], pre[0]) // pre[0] <- pre[1]
		in_degree[pre[0]]++
	}

	queue := []int{}
	for i := 0; i < numCourses; i++ {
		if in_degree[i] == 0 {
			queue = append(queue, i)
		}
	}

	for len(queue) != 0 {
		u := queue[0]
		queue = queue[1:]
		res = append(res, u)
		for _, v := range graph[u] {
			in_degree[v]--
			if in_degree[v] == 0 {
				queue = append(queue, v)
			}
		}
	}

	if len(res) == numCourses {// 无环
		return res
	} else {
		return []int{} //有环
	}
}
```

```go
func findOrder(numCourses int, prerequisites [][]int) []int {
	graph := make([][]int, numCourses)
	in_degree := make([]int, numCourses)
	visited := make([]int, numCourses)
	res := []int{}
	var findCircle func(int) bool

	findCircle = func(node int) bool {
		if visited[node] == 1 {
			return true
		}
		if visited[node] == 2 {
			return false
		}
		visited[node] = 1
		for _, next := range graph[node] {
			if findCircle(next) {
				return true
			}
		}
		visited[node] = 2
		res = append(res, node)
		return false
	}

	for _, pre := range prerequisites {
		graph[pre[1]] = append(graph[pre[1]], pre[0]) // pre[0] <- pre[1]
		in_degree[pre[0]]++
	}
	for i := 0; i < numCourses; i++ {
		if visited[i] == 0 {
			if findCircle(i) {
				return []int{}
			}
		}
	}

	for i := 0; i < numCourses/2; i++ {
		res[i], res[numCourses-i-1] = res[numCourses-i-1], res[i] //反转
	}

	return res
}
```



## [739. 每日温度](https://leetcode-cn.com/problems/daily-temperatures/)

```go
func dailyTemperatures(T []int) []int {
	n := len(T)
	res := make([]int, n)
	stack := []int{}
	for i := 0; i < n; i++ {
		t := T[i]
		for len(stack) > 0 && T[stack[len(stack)-1]] < t {
			prev_index := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			res[prev_index] = i - prev_index
		}
		stack = append(stack, i)
	}
	return res
}
```

## [26. 删除有序数组中的重复项](https://leetcode-cn.com/problems/remove-duplicates-from-sorted-array/)

```go
func removeDuplicates(A []int) int {
	if len(A) == 0 {
		return 0
	}
	slow := 1
	for fast := 1; fast < len(A); fast++ {
		if A[fast] != A[fast-1] {
			A[slow] = A[fast]
			slow++
		}
	}
	return slow
}
```

```go
func removeDuplicates(A []int) int {
	count, n := 0, len(A)
	for i := 1; i < n; i++ {
		if A[i] == A[i-1] {
			count++
		} else {
			A[i-count] = A[i]
		}
	}
	return n - count
}
```

## [287. 寻找重复数](https://leetcode-cn.com/problems/find-the-duplicate-number/)

### 方法一：快慢指针

```go
func findDuplicate(nums []int) int {
	slow := nums[0]
	fast := nums[nums[0]]
	for slow != fast {
		slow = nums[slow]
		fast = nums[nums[fast]]
	}
	fast = 0
	for fast != slow {
		fast = nums[fast]
		slow = nums[slow]
	}
	return slow
}
```
复杂度分析

- 时间复杂度：O(n)，其中 n 为 nums 数组的长度。

- 空间复杂度：O(1)。



### 方法二：二分查找

```go
func findDuplicate(nums []int) int {
	res, n := -1, len(nums)
	low, high := 0, n-1
	for low <= high {
		mid := low + (high-low)>>1
		cnt := 0
		for i := 0; i < n; i++ {
			if nums[i] <= mid {
				cnt++
			}
		}
		if cnt <= mid {
			low = mid + 1
		} else {
			high = mid - 1
			res = mid
		}
	}
	return res
}
```

```go
func findDuplicate(nums []int) int {
	low, high := 0, len(nums)-1
	for low < high {
		mid, count := low+(high-low)>>1, 0
		for _, num := range nums {
			if num <= mid {
				count++
			}
		}
		if count <= mid {
			low = mid + 1
		} else {
			high = mid
		}
	}
	return low
}
```

复杂度分析

- 时间复杂度：O(nlogn)，其中 n 为 nums 数组的长度。二分查找最多需要二分 O(logn) 次，每次判断的时候需要 O(n) 遍历 nums 数组求解小于等于 mid 的数的个数，因此总时间复杂度为 O(nlogn)。

- 空间复杂度：O(1)。我们只需要常数空间存放若干变量。


### 方法三：排序

```go
func findDuplicate(nums []int) int {
	sort.Ints(nums)
	for i := 0; i < len(nums); i++ {
		if nums[i] == nums[i+1] {
			return nums[i]
		}
	}
	return -1
}
```

复杂度分析

- 时间复杂度：O(nlogn)，其中 n 为 nums 数组的长度。

- 空间复杂度：O(1)。我们只需要常数空间存放若干变量。


### 方法四：哈希

```go
func findDuplicate(nums []int) int {
	hash := make(map[int]bool, len(nums))
	for _, num := range nums {
		if hash[num] {
			return num
		}
		hash[num] = true
	}
	return -1
}
```

复杂度分析

- 时间复杂度：O(n)，其中 n 为 nums 数组的长度。

- 空间复杂度：O(n)。


## [11. 盛最多水的容器](https://leetcode-cn.com/problems/container-with-most-water/)

```go
func maxArea(height []int) int {
	max_area := 0
	l, r := 0, len(height)-1
	for l < r {
		max_area = max(max_area, min(height[l], height[r])*(r-l))
		if height[l] < height[r] { //移动较小指针
			l++
		} else {
			r--
		}
	}
	return max_area
}
func max(x, y int) int {
	if x > y {
		return x
	}
	return y
}
func min(x, y int) int {
	if x < y {
		return x
	}
	return y
}
```

## [560. 和为K的子数组](https://leetcode-cn.com/problems/subarray-sum-equals-k/)

![截屏2021-05-30 15.45.51.png](http://ww1.sinaimg.cn/large/007daNw2ly1gr0i1voj83j31fm0oggqk.jpg)

```go
func subarraySum(nums []int, k int) int {
	pre, count := 0, 0
	m := map[int]int{0: 1}
	for i := 0; i < len(nums); i++ {
		pre += nums[i]
		if _, ok := m[pre-k]; ok {
			count += m[pre-k]
		}
		m[pre]++
	}
	return count
}
```

```go
func subarraySum(nums []int, k int) int {
	preSum, count := 0, 0
	hash := map[int]int{0: 1}
	for i := 0; i < len(nums); i++ {
		preSum += nums[i]
		if hash[preSum-k] > 0 {
			count += hash[preSum-k]
		}
		hash[preSum] += 1
	}
	return count
}
```


```go
func subarraySum(nums []int, k int) int {
	count := 0
	for i := 0; i < len(nums); i++ {
		sum := 0
		for j := i; j < len(nums); j++ {
			sum += nums[j]
			if sum == k {
				count++
			}
		}
	}
	return count
}
```

## [443. 压缩字符串](https://leetcode-cn.com/problems/string-compression/)


```go

```


## [50. Pow(x, n)](https://leetcode-cn.com/problems/powx-n/)



```go
func myPow(x float64, n int) float64 {
	if n >= 0 {
		return quickMul(x, n)
	}
	return 1.0 / quickMul(x, -n)
}
func quickMul(x float64, n int) float64 {
	res := 1.0
	for n > 0 {
		if n%2 == 1 {
			res *= x
		}
		x *= x
		n /= 2
	}
	return res
}
```

```go
func myPow(x float64, n int) float64 {
	if x == 0.0 {
		return 0.0
	}
	res := 1.0
	if n < 0 {
		x, n = 1/x, -n
	}
	for n != 0 {
		if n&1 == 1 { // 奇数: n%2 == 1 判断 n 二进制最右一位是否为 1
			res *= x
		}
		x *= x
		n >>= 1 // n = n/2
	}
	return res
}
```

复杂度分析

- 时间复杂度：O(logn)，即为对 n 进行二进制拆分的时间复杂度。

- 空间复杂度：O(1)。


```go
func myPow(x float64, n int) float64 {
	if n >= 0 {
		return quickMul(x, n)
	}
	return 1.0 / quickMul(x, -n)
}
func quickMul(x float64, n int) float64 {
	if n == 0 {
		return 1.0
	}
	y := quickMul(x, n/2)
	if n%2 == 1 {
		return x * y * y
	}
	return y * y
}
```

复杂度分析

- 时间复杂度：O(logn)，即为递归的层数。

- 空间复杂度：O(logn)，即为递归的层数。这是由于递归的函数调用会使用栈空间。



## [补充题2. 圆环回原点问题](https://mp.weixin.qq.com/s/VnGFEWHeD3nh1n9JSDkVUg)



