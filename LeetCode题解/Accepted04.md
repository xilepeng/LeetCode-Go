

1. [22. 括号生成](#22-括号生成)
2. [98. 验证二叉搜索树](#98-验证二叉搜索树)
3. [543. 二叉树的直径](#543-二叉树的直径)
4. [470. 用 Rand7() 实现 Rand10()](#470-用-rand7-实现-rand10)
5. [64. 最小路径和](#64-最小路径和)
6. [718. 最长重复子数组](#718-最长重复子数组)
7. [78. 子集](#78-子集)
8. [112. 路径总和](#112-路径总和)
9. [48. 旋转图像](#48-旋转图像)
10. [234. 回文链表](#234-回文链表)
11. [169. 多数元素](#169-多数元素)
12. [226. 翻转二叉树](#226-翻转二叉树)
13. [101. 对称二叉树](#101-对称二叉树)
14. [34. 在排序数组中查找元素的第一个和最后一个位置](#34-在排序数组中查找元素的第一个和最后一个位置)
15. [83. 删除排序链表中的重复元素](#83-删除排序链表中的重复元素)
16. [165. 比较版本号](#165-比较版本号)
17. [240. 搜索二维矩阵 II](#240-搜索二维矩阵-ii)
18. [136. 只出现一次的数字](#136-只出现一次的数字)
19. [153. 寻找旋转排序数组中的最小值](#153-寻找旋转排序数组中的最小值)
20. [34. 在排序数组中查找元素的第一个和最后一个位置](#34-在排序数组中查找元素的第一个和最后一个位置-1)
21. [39. 组合总和](#39-组合总和)
22. [62. 不同路径](#62-不同路径)



<!-- 

[22. 括号生成](https://leetcode-cn.com/problems/generate-parentheses/)

[98. 验证二叉搜索树](https://leetcode-cn.com/problems/validate-binary-search-tree/)

[543. 二叉树的直径](https://leetcode-cn.com/problems/diameter-of-binary-tree/)

[470. 用 Rand7() 实现 Rand10()](https://leetcode-cn.com/problems/implement-rand10-using-rand7/)

[64. 最小路径和](https://leetcode-cn.com/problems/minimum-path-sum/)

[718. 最长重复子数组](https://leetcode-cn.com/problems/maximum-length-of-repeated-subarray/)

[78. 子集](https://leetcode-cn.com/problems/subsets/)

[112. 路径总和](https://leetcode-cn.com/problems/path-sum/)

[48. 旋转图像](https://leetcode-cn.com/problems/rotate-image/)

[234. 回文链表](https://leetcode-cn.com/problems/palindrome-linked-list/) -->


------




## [22. 括号生成](https://leetcode-cn.com/problems/generate-parentheses/)


```go
func generateParenthesis(n int) (res []string) {
	var dfs func(int, int, string)
	dfs = func(left, right int, path string) {
		if len(path) == 2*n { // 一个合法解已生成
			res = append(res, path) // 加入解集
			return
		}
		if left > 0 { // 只要左括号有剩余,选左括号
			dfs(left-1, right, path+"(")
		}
		if right > left { // 右括号数量大于左括号，选右括号
			dfs(left, right-1, path+")")
		}
	}
	dfs(n, n, "")
	return
}
```



[参考](https://leetcode-cn.com/problems/generate-parentheses/solution/shou-hua-tu-jie-gua-hao-sheng-cheng-hui-su-suan-fa/)


**方法二：中序遍历**
思路和算法

基于方法一中提及的性质，我们可以进一步知道二叉搜索树「中序遍历」得到的值构成的序列一定是升序的，这启示我们在中序遍历的时候实时检查当前节点的值是否大于前一个中序遍历到的节点的值即可。如果均大于说明这个序列是升序的，整棵树是二叉搜索树，否则不是

```go
func isValidBST(root *TreeNode) bool {
	stack := []*TreeNode{}
	inorder := math.MinInt64
	for len(stack) > 0 || root != nil {
		for root != nil {
			stack = append(stack, root)
			root = root.Left
		}
		root = stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		if root.Val <= inorder {
			return false
		}
		inorder = root.Val
		root = root.Right
	}
	return true
}
```

解法二，把 BST 按照左中右的顺序输出到数组中，如果是 BST，则数组中的数字是从小到大有序的，如果出现逆序就不是 BST

```go
func isValidBST(root *TreeNode) bool {
	nums := []int{}

	var inorder func(*TreeNode)
	inorder = func(root *TreeNode) {
		if root == nil {
			return
		}
		inorder(root.Left)
		nums = append(nums, root.Val)
		inorder(root.Right)
	}
	inorder(root)

	for i := 1; i < len(nums); i++ {
		if nums[i-1] >= nums[i] {
			return false
		}
	}
	return true
}
```

复杂度分析

- 时间复杂度 : O(n)，其中 n 为二叉树的节点个数。二叉树的每个节点最多被访问一次，因此时间复杂度为 O(n)。
- 空间复杂度 : O(n)，其中 n 为二叉树的节点个数。栈最多存储 n 个节点，因此需要额外的 O(n) 的空间。






## [98. 验证二叉搜索树](https://leetcode-cn.com/problems/validate-binary-search-tree/)

**方法一: 递归**

思路和算法

解法一，直接按照定义比较大小，比 root 节点小的都在左边，比 root 节点大的都在右边

```go
/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */
func isValidBST(root *TreeNode) bool {
	return dfs(root, math.MinInt64, math.MaxInt64)
}
func dfs(node *TreeNode, lower, upper int) bool {
	if node == nil { // 只有一个结点
		return true
	}
	if node.Val <= lower || node.Val >= upper {
		return false // 越界
	} // 递归检查左右子树
	return dfs(node.Left, lower, node.Val) && dfs(node.Right, node.Val, upper)
}
```



```go
func isValidBST(root *TreeNode) bool {
	return dfs(root, -1<<63, 1<<63-1)
}
func dfs(root *TreeNode, lower, upper int) bool {
	return root == nil || root.Val > lower && root.Val < upper &&
		dfs(root.Left, lower, root.Val) &&
		dfs(root.Right, root.Val, upper)
}
```




复杂度分析

时间复杂度 : O(n)，其中 n 为二叉树的节点个数。在递归调用的时候二叉树的每个节点最多被访问一次，因此时间复杂度为 O(n)。

空间复杂度 : O(n)，其中 n 为二叉树的节点个数。递归函数在递归过程中需要为每一层递归函数分配栈空间，所以这里需要额外的空间且该空间取决于递归的深度，即二叉树的高度。最坏情况下二叉树为一条链，树的高度为 n ，递归最深达到 n 层，故最坏情况下空间复杂度为 O(n) 。











[参考](https://leetcode-cn.com/problems/validate-binary-search-tree/solution/yan-zheng-er-cha-sou-suo-shu-by-leetcode-solution/)




## [543. 二叉树的直径](https://leetcode-cn.com/problems/diameter-of-binary-tree/)

```go
/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */
func diameterOfBinaryTree(root *TreeNode) int {
	res := 1
	var depth func(*TreeNode) int

	depth = func(node *TreeNode) int {
		if node == nil {
			return 0
		}
		left, right := depth(node.Left), depth(node.Right) // 左右子树最大深度
		res = max(res, left+right+1)                       // max(最大直径，当前节点的直径)
		return max(left, right) + 1                        // 返回该节点为根的子树的深度
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






## [470. 用 Rand7() 实现 Rand10()](https://leetcode-cn.com/problems/implement-rand10-using-rand7/)


```go
[1,7]
0  t =[1,7]
7  t = [1,49]  

t = 1    min = 1
...
t = 40  (40-1)%10 + 1 = 10
```

```go
func rand10() int {
	t := (rand7()-1)*7 + rand7() //t = [1, 49]
	if t > 40 {
		return rand10()
	}
	return (t-1)%10 + 1          // [1, 10]
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




## [64. 最小路径和](https://leetcode-cn.com/problems/minimum-path-sum/)



**方法一：原地 DP，无辅助空间**

```go
func minPathSum(grid [][]int) int {
	m, n := len(grid), len(grid[0]) // m 行 n 列
	for i := 1; i < m; i++ {
		grid[i][0] += grid[i-1][0] // 第0列 累加和
	}
	for j := 1; j < n; j++ {
		grid[0][j] += grid[0][j-1] // 第0行 累加和
	}
	for i := 1; i < m; i++ {
		for j := 1; j < n; j++ {
			grid[i][j] += min(grid[i-1][j], grid[i][j-1])
		}
	}
	return grid[m-1][n-1]
}
func min(x, y int) int {
	if x < y {
		return x
	}
	return y
}
```


复杂度分析

- 时间复杂度：O(mn)，其中 m 和 n 分别是网格的行数和列数。需要对整个网格遍历一次，计算 dp 的每个元素的值。

- 空间复杂度：O(1)


[参考](https://leetcode-cn.com/problems/minimum-path-sum/solution/zui-xiao-lu-jing-he-by-leetcode-solution/)



**方法二：动态规划**















## [718. 最长重复子数组](https://leetcode-cn.com/problems/maximum-length-of-repeated-subarray/)


**方法一：暴力**


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




**方法二：动态规划**


**动态规划法：**
- A 、B数组各抽出一个子数组，单看它们的末尾项，如果它们俩不一样，则公共子数组肯定不包括它们俩。
- 如果它们俩一样，则要考虑它们俩前面的子数组「能为它们俩提供多大的公共长度」。
	- 如果它们俩的前缀数组的「末尾项」不相同，由于子数组的连续性，前缀数组不能为它们俩提供公共长度
	- 如果它们俩的前缀数组的「末尾项」相同，则可以为它们俩提供公共长度：
	至于提供多长的公共长度？这又取决于前缀数组的末尾项是否相同……
**加上注释再讲一遍**
- A 、B数组各抽出一个子数组，单看它们的末尾项，如果它们俩不一样——以它们俩为末尾项形成的公共子数组的长度为0：dp[i][j] = 0
- 如果它们俩一样，以它们俩为末尾项的公共子数组，长度保底为1——dp[i][j]至少为 1，要考虑它们俩的前缀数组——dp[i-1][j-1]能为它们俩提供多大的公共长度
1. 如果它们俩的前缀数组的「末尾项」不相同，前缀数组提供的公共长度为 0——dp[i-1][j-1] = 0
	- 以它们俩为末尾项的公共子数组的长度——dp[i][j] = 1
2. 如果它们俩的前缀数组的「末尾项」相同
	- 前缀部分能提供的公共长度——dp[i-1][j-1]，它至少为 1
	- 以它们俩为末尾项的公共子数组的长度 dp[i][j] = dp[i-1][j-1] + 1
- 题目求：最长公共子数组的长度。不同的公共子数组的末尾项不一样。我们考察不同末尾项的公共子数组，找出最长的那个。（注意下图的最下方的一句话）



![1.png](http://ww1.sinaimg.cn/large/007daNw2ly1gpfh1ulfu2j31go0wkn6i.jpg)

**状态转移方程**
- dp[i][j] ：长度为i，末尾项为A[i-1]的子数组，与长度为j，末尾项为B[j-1]的子数组，二者的最大公共后缀子数组长度。
	如果 A[i-1] != B[j-1]， 有 dp[i][j] = 0
	如果 A[i-1] == B[j-1] ， 有 dp[i][j] = dp[i-1][j-1] + 1
- base case：如果i==0||j==0，则二者没有公共部分，dp[i][j]=0
- 最长公共子数组以哪一项为末尾项都有可能，求出每个 dp[i][j]，找出最大值。


![2.png](http://ww1.sinaimg.cn/large/007daNw2ly1gpfh22j20lj31880lktd2.jpg)


**代码**
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


**降维优化**
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

**方法三：动态规划**



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











## [78. 子集](https://leetcode-cn.com/problems/subsets/)


**方法一：位运算**

![截屏2021-04-17 11.45.32.png](http://ww1.sinaimg.cn/large/007daNw2ly1gpmlgt89mej31120o2wgb.jpg)

```go
func subsets(nums []int) [][]int {
	res, n := [][]int{}, len(nums)
    //1<<3 二进制：1000 十进制：1*2^n=8
	for i := 0; i < 1<<n; i++ { // i 从 000 到 111 
		tmp := []int{}
		for j := 0; j < n; j++ {
			if i>>j&1 == 1 { // i 的第 j 位是否为1
				tmp = append(tmp, nums[j])
			}
		}
		res = append(res, tmp)
	}
	return res
}
```

**方法一：迭代法实现子集枚举**
思路与算法

记原序列中元素的总数为 n。原序列中的每个数字 ai 的状态可能有两种，即「在子集中」和「不在子集中」。我们用 1 表示「在子集中」，0 表示不在子集中，那么每一个子集可以对应一个长度为 n 的 0/1 序列，第 i 位表示 ai 是否在子集中。
例如，n=3 ，a={5,2,9} 时：

![截屏2021-04-17 14.57.34.png](http://ww1.sinaimg.cn/large/007daNw2ly1gpmr0nopx0j316u0judho.jpg)

可以发现 0/1 序列对应的二进制数正好从 0 到 2^n - 1。我们可以枚举 mask∈[0, 2^n - 1]，mask 的二进制表示是一个 0/1 序列，我们可以按照这个 0/1 序列在原集合当中取数。当我们枚举完所有 2^n 个 mask，我们也就能构造出所有的子集。

```go
func subsets(nums []int) [][]int {
	res, n := [][]int{}, len(nums)
	for mask := 0; mask < 1<<n; mask++ {
		set := []int{}
		for i, v := range nums {
			if mask>>i&1 == 1 {
				set = append(set, v)
			}
		}
		res = append(res, append([]int(nil), set...))
	}
	return res
}
```

复杂度分析

- 时间复杂度：O(n×2^n)。一共 2^n 个状态，每种状态需要 O(n) 的时间来构造子集。

- 空间复杂度：O(n)。即构造子集使用的临时数组 t 的空间代价。



**方法二：递归法实现子集枚举**

**思路 1**
- 单看每个元素，都有两种选择：选入子集，或不选入子集。
- 比如[1,2,3]，先看1，选1或不选1，都会再看2，选2或不选2，以此类推。
- 考察当前枚举的数，基于选它而继续，是一个递归分支；基于不选它而继续，又是一个分支。

![1.png](http://ww1.sinaimg.cn/large/007daNw2ly1gpms453jltj31fw0jjq6r.jpg)

- 用索引index代表当前递归考察的数字nums[index]。
- 当index越界时，所有数字考察完，得到一个解，位于递归树的底部，把它加入解集，结束当前递归分支。

**为什么要回溯？**
- 因为不是找到一个子集就完事。
- 找到一个子集，结束递归，要撤销当前的选择，回到选择前的状态，做另一个选择——不选当前的数，基于不选，往下递归，继续生成子集。
- 回退到上一步，才能在包含解的空间树中把路走全，回溯出所有的解。

![2.png](http://ww1.sinaimg.cn/large/007daNw2ly1gpms4bgnnkj31a60lc0w8.jpg)


```go
func subsets(nums []int) [][]int {
	res, set := [][]int{}, []int{}
	var dfs func(int)

	dfs = func(i int) {
		if i == len(nums) { // 指针越界
			res = append(res, append([]int(nil), set...)) // 加入解集
			return                                        // 结束当前的递归
		}
		set = append(set, nums[i]) //选择这个数
		dfs(i + 1)                 // 基于该选择，继续往下递归，考察下一个数
		set = set[:len(set)-1]     // 上面的递归结束，撤销该选择
		dfs(i + 1)                 // 不选这个数，继续往下递归，考察下一个数
	}

	dfs(0)
	return res
}
```

**思路2**

刚才的思路是：逐个考察数字，每个数都选或不选。等到递归结束时，把集合加入解集。
换一种思路：在执行子递归之前，加入解集，即，在递归压栈前 “做事情”。

![3.png](http://ww1.sinaimg.cn/large/007daNw2ly1gpms4h4i9rj31070h1mz8.jpg)

- 用 for 枚举出当前可选的数，比如选第一个数时：1、2、3 可选。
1. 如果第一个数选 1，选第二个数，2、3 可选；
2. 如果第一个数选 2，选第二个数，只有 3 可选（不能选1，产生重复组合）
3. 如果第一个数选 3，没有第二个数可选
- 每次传入子递归的 index 是：当前你选的数的索引+1当前你选的数的索引+1。
- 每次递归枚举的选项变少，一直递归到没有可选的数字，进入不了for循环，落入不了递归，整个DFS结束。
- 可见我们没有显式地设置递归的出口，而是通过控制循环的起点，使得最后递归自然结束。

```go
func subsets(nums []int) [][]int {
	res, set := [][]int{}, []int{}
	var dfs func(int)

	dfs = func(i int) {
		res = append(res, append([]int(nil), set...)) // 调用子递归前，加入解集
		for j := i; j < len(nums); j++ {              // 枚举出所有可选的数
			set = append(set, nums[j]) // 选这个数
			dfs(j + 1)                 // 基于选这个数，继续递归，传入的j+1，不是i+1
			set = set[:len(set)-1]     // 撤销选这个数
		}
	}

	dfs(0)
	return res
}
```







## [112. 路径总和](https://leetcode-cn.com/problems/path-sum/)

**方法一：递归**

```go
func hasPathSum(root *TreeNode, sum int) bool {
	if root == nil {
		return false // 遍历到null节点
	}
	if root.Left == nil && root.Right == nil { // 遍历到叶子节点
		return sum-root.Val == 0 // 如果满足这个就返回true。否则返回false
	} // 当前递归问题 拆解成 两个子树的问题，其中一个true了就行
	return hasPathSum(root.Left, sum-root.Val) || hasPathSum(root.Right, sum-root.Val)
}
```






## [48. 旋转图像](https://leetcode-cn.com/problems/rotate-image/)

**方法一：用翻转代替旋转**

![截屏2021-04-20 17.50.27.png](http://ww1.sinaimg.cn/large/007daNw2ly1gpqcvh0n5pj314w0mkadk.jpg)

![截屏2021-04-20 17.50.41.png](http://ww1.sinaimg.cn/large/007daNw2ly1gpqcvp2lc4j318e0o60xd.jpg)

```go
func rotate(matrix [][]int) {
	n := len(matrix)
	// 水平翻转
	for i := 0; i < n/2; i++ {
		matrix[i], matrix[n-1-i] = matrix[n-1-i], matrix[i]
	}
	// 主对角线翻转
	for i := 0; i < n; i++ {
		for j := 0; j < i; j++ {
			matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
		}
	}
}
```











## [234. 回文链表](https://leetcode-cn.com/problems/palindrome-linked-list/)

**方法一：转成数组**
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




**方法二：快慢指针**
快慢指针，起初都指向表头，快指针一次走两步，慢指针一次走一步，遍历结束时：

- 要么，slow 正好指向中间两个结点的后一个。
- 要么，slow 正好指向中间结点。
用 prev 保存 slow 的前一个结点，通过prev.next = null断成两个链表。

将后半段链表翻转，和前半段从头比对。空间复杂度降为O(1)。

![1.png](http://ww1.sinaimg.cn/large/007daNw2ly1gpecx13c3rj30z20fcjv7.jpg)

**如何翻转单链表**
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













---






[169. 多数元素](https://leetcode-cn.com/problems/majority-element/)

[876. 链表的中间结点](https://leetcode-cn.com/problems/middle-of-the-linked-list/) 补充

[226. 翻转二叉树](https://leetcode-cn.com/problems/invert-binary-tree/)

[101. 对称二叉树](https://leetcode-cn.com/problems/symmetric-tree/)

[83. 删除排序链表中的重复元素](https://leetcode-cn.com/problems/remove-duplicates-from-sorted-list/)

[113. 路径总和 II](https://leetcode-cn.com/problems/path-sum-ii/) 补充

[165. 比较版本号](https://leetcode-cn.com/problems/compare-version-numbers/)



[240. 搜索二维矩阵 II](https://leetcode-cn.com/problems/search-a-2d-matrix-ii/)



[153. 寻找旋转排序数组中的最小值](https://leetcode-cn.com/problems/find-minimum-in-rotated-sorted-array/)

[39. 组合总和](https://leetcode-cn.com/problems/combination-sum/)

[136. 只出现一次的数字](https://leetcode-cn.com/problems/single-number/)

[34. 在排序数组中查找元素的第一个和最后一个位置](https://leetcode-cn.com/problems/find-first-and-last-position-of-element-in-sorted-array/)

[958. 二叉树的完全性检验](https://leetcode-cn.com/problems/check-completeness-of-a-binary-tree/)

[62. 不同路径](https://leetcode-cn.com/problems/unique-paths/)

[34. 在排序数组中查找元素的第一个和最后一个位置](https://leetcode-cn.com/problems/find-first-and-last-position-of-element-in-sorted-array/)



---







## [169. 多数元素](https://leetcode-cn.com/problems/majority-element/)

**方法五：Boyer-Moore 投票算法**
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






## [226. 翻转二叉树](https://leetcode-cn.com/problems/invert-binary-tree/)

**方法一：dfs 递归**

**递归思路1**

我们从根节点开始，递归地对树进行遍历，并从叶子结点先开始翻转。如果当前遍历到的节点 root 的左右两棵子树都已经翻转，那么我们只需要交换两棵子树的位置，即可完成以 root 为根节点的整棵子树的翻转。

*思路*
一个二叉树，怎么才算翻转了？

它的左右子树要交换，并且左右子树内部的所有子树，都要进行左右子树的交换。

![1.png](http://ww1.sinaimg.cn/large/007daNw2ly1gpmz43wk4yj31er0fw0w1.jpg)


每个子树的根节点都说：先交换我的左右子树吧。那么递归就会先压栈压到底。然后才做交换。
即，位于底部的、左右孩子都是 null 的子树，先被翻转。
随着递归向上返回，子树一个个被翻转……整棵树翻转好了。
问题是在递归出栈时解决的。

```go
func invertTree(root *TreeNode) *TreeNode {
	if root == nil {
		return nil
	}
	invertTree(root.Left)
	invertTree(root.Right)
	root.Left, root.Right = root.Right, root.Left
	return root
}
```

**递归思路 2**

![2.png](http://ww1.sinaimg.cn/large/007daNw2ly1gpmz4kjl1jj31fu0gh77e.jpg)

思路变了：先 “做事”——先交换左右子树，它们内部的子树还没翻转——丢给递归去做。
把交换的操作，放在递归子树之前。
问题是在递归压栈前被解决的。

```go
func invertTree(root *TreeNode) *TreeNode {
	if root == nil {
		return nil
	}
	root.Left, root.Right = root.Right, root.Left
	invertTree(root.Left)
	invertTree(root.Right)
	return root
}
```

复杂度分析

- 时间复杂度：O(N)，其中 N 为二叉树节点的数目。我们会遍历二叉树中的每一个节点，对每个节点而言，我们在常数时间内交换其两棵子树。

- 空间复杂度：O(N)。使用的空间由递归栈的深度决定，它等于当前节点在二叉树中的高度。在平均情况下，二叉树的高度与节点个数为对数关系，即 O(logN)。而在最坏情况下，树形成链状，空间复杂度为 O(N)。



**总结**
两种分别是后序遍历和前序遍历。都是基于DFS，都是先遍历根节点、再遍历左子树、再右子树。
唯一的区别是：
前序遍历：将「处理当前节点」放到「递归左子树」之前。
后序遍历：将「处理当前节点」放到「递归右子树」之后。

这个「处理当前节点」，就是交换左右子树 ，就是解决问题的代码：

```go
root.Left, root.Right = root.Right, root.Left
```

递归只是帮你遍历这棵树，核心还是解决问题的代码，递归把它应用到每个子树上，解决每个子问题，最后解决整个问题。

**方法二：BFS **

用层序遍历的方式去遍历二叉树。

根节点先入列，然后出列，出列就 “做事”，交换它的左右子节点（左右子树）。
并让左右子节点入列，往后，这些子节点出列，也被翻转。
直到队列为空，就遍历完所有的节点，翻转了所有子树。

解决问题的代码放在节点出列时。


```go
func invertTree(root *TreeNode) *TreeNode {
	if root == nil {
		return nil
	}
	q := []*TreeNode{root}
	for len(q) > 0 {
		cur := q[0]
		q = q[1:len(q)]
		cur.Left, cur.Right = cur.Right, cur.Left
		if cur.Left != nil {
			q = append(q, cur.Left)
		}
		if cur.Right != nil {
			q = append(q, cur.Right)
		}
	}
	return root
}
```





## [101. 对称二叉树](https://leetcode-cn.com/problems/symmetric-tree/)

**方法一：递归**
```go
/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */
func isSymmetric(root *TreeNode) bool {
	return isMirror(root, root)
}
func isMirror(left, right *TreeNode) bool {
	if left == nil && right == nil {
		return true
	}
	if left == nil || right == nil {
		return false
	}
	return left.Val == right.Val && isMirror(left.Left, right.Right) && isMirror(left.Right, right.Left)
}
```

**方法二：迭代**
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







## [34. 在排序数组中查找元素的第一个和最后一个位置](https://leetcode-cn.com/problems/find-first-and-last-position-of-element-in-sorted-array/)

**方法一：二分查找**

**解题思路**

- 给出一个有序数组 nums 和一个数 target，要求在数组中找到第一个和这个元素相等的元素下标，最后一个和这个元素相等的元素下标。

- 这一题是经典的二分搜索变种题。二分搜索有 4 大基础变种题：

	1. 查找第一个值等于给定值的元素
	2. 查找最后一个值等于给定值的元素
	3. 查找第一个大于等于给定值的元素
	4. 查找最后一个小于等于给定值的元素
这一题的解题思路可以分别利用变种 1 和变种 2 的解法就可以做出此题。或者用一次变种 1 的方法，然后循环往后找到最后一个与给定值相等的元素。不过后者这种方法可能会使时间复杂度下降到 O(n)，因为有可能数组中 n 个元素都和给定元素相同。(4 大基础变种的实现见代码)

```go
func searchRange(nums []int, target int) []int {
	return []int{searchFirstEqualElement(nums, target), searchLastEqualElement(nums, target)}
}

// 二分查找第一个与 target 相等的元素，时间复杂度 O(logn)
func searchFirstEqualElement(nums []int, target int) int {
	left, right := 0, len(nums)-1
	for left <= right {
		mid := left + (right-left)>>1
		if nums[mid] < target {
			left = mid + 1
		} else if nums[mid] > target {
			right = mid - 1
		} else {
			if mid == 0 || nums[mid-1] != target { // 找到第一个与 target 相等的元素
				return mid
			}
			right = mid - 1
		}
	}
	return -1
}

// 二分查找最后一个与 target 相等的元素，时间复杂度 O(logn)
func searchLastEqualElement(nums []int, target int) int {
	left, right := 0, len(nums)-1
	for left <= right {
		mid := left + (right-left)>>1
		if nums[mid] < target {
			left = mid + 1
		} else if nums[mid] > target {
			right = mid - 1
		} else {
			if mid == len(nums)-1 || nums[mid+1] != target { // 找到最后一个与 target 相等的元素
				return mid
			}
			left = mid + 1
		}
	}
	return -1
}

// 二分查找第一个大于等于 target 的元素，时间复杂度 O(logn)
func searchFirstGreaterElement(nums []int, target int) int {
	left, right := 0, len(nums)-1
	for left <= right {
		mid := left + (right-left)>>1
		if nums[mid] >= target {
			if mid == 0 || nums[mid-1] < target { // 找到第一个大于等于 target 的元素
				return mid
			}
			right = mid - 1
		} else {
			left = mid + 1
		}
	}
	return -1
}

// 二分查找最后一个小于等于 target 的元素，时间复杂度 O(logn)
func searchLastLessElement(nums []int, target int) int {
	left, right := 0, len(nums)-1
	for left <= right {
		mid := left + (right-left)>>1
		if nums[mid] <= target {
			if mid == len(nums)-1 || nums[mid+1] > target { // 找到最后一个小于等于 target 的元素
				return mid
			}
			left = mid + 1
		} else {
			right = mid - 1
		}
	}
	return -1
}
```

**方法二：二分查找**

```go
func searchRange(nums []int, target int) []int {
	leftmost := sort.SearchInts(nums, target)
	if leftmost == len(nums) || nums[leftmost] != target {
		return []int{-1, -1}
	}
	rightmost := sort.SearchInts(nums, target+1) - 1
	return []int{leftmost, rightmost}
}
```





## [83. 删除排序链表中的重复元素](https://leetcode-cn.com/problems/remove-duplicates-from-sorted-list/)

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




## [165. 比较版本号](https://leetcode-cn.com/problems/compare-version-numbers/)

```go
func compareVersion(s1 string, s2 string) int {
	i, j := 0, 0
	for i < len(s1) || j < len(s2) {
		a, b := "", ""
		for i < len(s1) && s1[i] != '.' {
			a += string(s1[i])
			i++
		}
		for j < len(s2) && s2[j] != '.' {
			b += string(s2[j])
			j++
		}
		x, _ := strconv.Atoi(a) //string 转 int
		y, _ := strconv.Atoi(b)
		if x > y {
			return 1
		} else if x < y {
			return -1
		}
		i++
		j++
	}
	return 0
}
```

```go
strconv.Atoi()函数用于将字符串类型的整数转换为int类型，函数签名如下。

func Atoi(s string) (i int, err error)


strconv.Itoa()函数用于将int类型数据转换为对应的字符串表示，具体的函数签名如下。

func Itoa(i int) string
```








## [240. 搜索二维矩阵 II](https://leetcode-cn.com/problems/search-a-2d-matrix-ii/)
![](https://pic.leetcode-cn.com/Figures/240/Slide3.PNG)

**方法一：模拟** 

1. 从右上角开始搜索


```go
func searchMatrix(matrix [][]int, target int) bool {
	row, col := 0, len(matrix[0])-1 // 从右上角开始遍历
	for row < len(matrix) && col >= 0 {
		if matrix[row][col] < target { // 小于目标值向下搜索
			row++
		} else if matrix[row][col] > target { // 大于目标值向左搜索
			col--
		} else {
			return true // 等于目标值返回true
		}
	}
	return false
}
```

2. 从左下角开始搜索



```go
func searchMatrix(matrix [][]int, target int) bool {
	row, col := len(matrix)-1, 0 // 从左下角开始遍历
	for row >= 0 && col < len(matrix[0]) {
		if matrix[row][col] < target { // 小于目标值向右搜索
			col++
		} else if matrix[row][col] > target { // 大于目标值向上搜索
			row--
		} else {
			return true // 等于目标值返回true
		}
	}
	return false
}
```

复杂度分析

- 时间复杂度：O(n+m)。
时间复杂度分析的关键是注意到在每次迭代（我们不返回 true）时，行或列都会精确地递减/递增一次。由于行只能减少 m 次，而列只能增加 n 次，因此在导致 for 循环终止之前，循环不能运行超过 n+m 次。因为所有其他的工作都是常数，所以总的时间复杂度在矩阵维数之和中是线性的。
- 空间复杂度：O(1)，因为这种方法只处理几个指针，所以它的内存占用是恒定的。

**方法二：二分法搜索**

```go
func searchMatrix(matrix [][]int, target int) bool {
	for _, row := range matrix {
		low, high := 0, len(matrix[0])-1
		for low <= high {
			mid := low + (high-low)>>1
			if target == row[mid] {
				return true
			} else if target < row[mid] {
				high = mid - 1
			} else {
				low = mid + 1
			}
		}
	}
	return false
}
```

复杂度分析

- 时间复杂度 O(n log n)
- 空间复杂度：O(1)




## [136. 只出现一次的数字](https://leetcode-cn.com/problems/single-number/)

方法一：位运算 

异或运算的作用
　　参与运算的两个值，如果两个相应bit位相同，则结果为0，否则为1。

　　即：

　　0^0 = 0，

　　1^0 = 1，

　　0^1 = 1，

　　1^1 = 0

　　按位异或的3个特点：

　　（1） 0^0=0，0^1=1  0异或任何数＝任何数

　　（2） 1^0=1，1^1=0  1异或任何数－任何数取反

　　（3） 任何数异或自己＝把自己置0

```go
func singleNumber(nums []int) int {
	res := 0
	for _, num := range nums {
		res ^= num
	}
	return res
}
```







## [153. 寻找旋转排序数组中的最小值](https://leetcode-cn.com/problems/find-minimum-in-rotated-sorted-array/)

**思路**
1. 有序数组分成了左右2个小的有序数组，而实际上要找的是右边有序数组的最小值
2. 如果中间值大于右边的最大值，说明中间值还在左边的小数组里，需要left向右移动
3. 否则就是中间值小于等于当前右边最大值，mid 已经在右边的小数组里了，但是至少说明了当前右边的right值不是最小值了或者不是唯一的最小值，需要慢慢向左移动一位。


```go
func findMin(nums []int) int {
	left, right := 0, len(nums)-1
	for left <= right {
		mid := left + (right-left)>>1
		if nums[mid] > nums[right] {
			left = mid + 1
		} else {
			right--
		}
	}
	return nums[left]
}
```

```go
func findMin(nums []int) int {
	left, right := 0, len(nums)-1
	for left < right {
		mid := left + (right-left)>>1
		if nums[mid] < nums[right] {
			right = mid
		} else {
			left = mid + 1
		}
	}
	return nums[left]
}
```



## [34. 在排序数组中查找元素的第一个和最后一个位置](https://leetcode-cn.com/problems/find-first-and-last-position-of-element-in-sorted-array/)



**方法一**

```go
func searchRange(nums []int, target int) []int {
	first, last := findFirst(nums, target), findLast(nums, target)
	return []int{first, last}
}
func findFirst(nums []int, target int) int {
	low, high := 0, len(nums)-1
	index := -1
	for low <= high {
		mid := low + (high-low)>>1
		if nums[mid] >= target {
			high = mid - 1
		} else {
			low = mid + 1
		}
		if nums[mid] == target {
			index = mid
		}
	}
	return index
}
func findLast(nums []int, target int) int {
	low, high := 0, len(nums)-1
	index := -1
	for low <= high {
		mid := low + (high-low)>>1
		if nums[mid] <= target {
			low = mid + 1
		} else {
			high = mid - 1
		}
		if nums[mid] == target {
			index = mid
		}
	}
	return index
}
```


**方法二：二分查找**

**解题思路** 
- 给出一个有序数组 nums 和一个数 target，要求在数组中找到第一个和这个元素相等的元素下标，最后一个和这个元素相等的元素下标。

- 这一题是经典的二分搜索变种题。二分搜索有 4 大基础变种题：

	1. 查找第一个值等于给定值的元素
	2. 查找最后一个值等于给定值的元素
	3. 查找第一个大于等于给定值的元素
	4. 查找最后一个小于等于给定值的元素
这一题的解题思路可以分别利用变种 1 和变种 2 的解法就可以做出此题。或者用一次变种 1 的方法，然后循环往后找到最后一个与给定值相等的元素。不过后者这种方法可能会使时间复杂度下降到 O(n)，因为有可能数组中 n 个元素都和给定元素相同。(4 大基础变种的实现见代码)

```go
func searchRange(nums []int, target int) []int {
	return []int{searchFirstEqualElement(nums, target), searchLastEqualElement(nums, target)}
}

// 二分查找第一个与 target 相等的元素，时间复杂度 O(logn)
func searchFirstEqualElement(nums []int, target int) int {
	left, right := 0, len(nums)-1
	for left <= right {
		mid := left + (right-left)>>1
		if nums[mid] < target {
			left = mid + 1
		} else if nums[mid] > target {
			right = mid - 1
		} else {
			if mid == 0 || nums[mid-1] != target { // 找到第一个与 target 相等的元素
				return mid
			}
			right = mid - 1
		}
	}
	return -1
}

// 二分查找最后一个与 target 相等的元素，时间复杂度 O(logn)
func searchLastEqualElement(nums []int, target int) int {
	left, right := 0, len(nums)-1
	for left <= right {
		mid := left + (right-left)>>1
		if nums[mid] < target {
			left = mid + 1
		} else if nums[mid] > target {
			right = mid - 1
		} else {
			if mid == len(nums)-1 || nums[mid+1] != target { // 找到最后一个与 target 相等的元素
				return mid
			}
			left = mid + 1
		}
	}
	return -1
}

// 二分查找第一个大于等于 target 的元素，时间复杂度 O(logn)
func searchFirstGreaterElement(nums []int, target int) int {
	left, right := 0, len(nums)-1
	for left <= right {
		mid := left + (right-left)>>1
		if nums[mid] >= target {
			if mid == 0 || nums[mid-1] < target { // 找到第一个大于等于 target 的元素
				return mid
			}
			right = mid - 1
		} else {
			left = mid + 1
		}
	}
	return -1
}

// 二分查找最后一个小于等于 target 的元素，时间复杂度 O(logn)
func searchLastLessElement(nums []int, target int) int {
	left, right := 0, len(nums)-1
	for left <= right {
		mid := left + (right-left)>>1
		if nums[mid] <= target {
			if mid == len(nums)-1 || nums[mid+1] > target { // 找到最后一个小于等于 target 的元素
				return mid
			}
			left = mid + 1
		} else {
			right = mid - 1
		}
	}
	return -1
}
```
**方法三：二分查找**

```go
func searchRange(nums []int, target int) []int {
	leftmost := sort.SearchInts(nums, target)
	if leftmost == len(nums) || nums[leftmost] != target {
		return []int{-1, -1}
	}
	rightmost := sort.SearchInts(nums, target+1) - 1
	return []int{leftmost, rightmost}
}
```




## [39. 组合总和](https://leetcode-cn.com/problems/combination-sum/)


**方法一：搜索回溯**

```go
func combinationSum(candidates []int, target int) [][]int {
	comb, res := []int{}, [][]int{}
	var dfs func(int, int)

	dfs = func(target int, idx int) {
		if idx == len(candidates) {
			return
		}
		if target == 0 {
			res = append(res, append([]int(nil), comb...))
			return
		}
		// 直接跳过
		dfs(target, idx+1)
		// 选择当前数
		if target-candidates[idx] >= 0 {
			comb = append(comb, candidates[idx])
			dfs(target-candidates[idx], idx)
			comb = comb[:len(comb)-1]
		}
	}
	dfs(target, 0)
	return res
}
```

[参考](https://leetcode.cn/problems/combination-sum/solution/zu-he-zong-he-by-leetcode-solution/)


**剪枝优化**

- ×：当前组合和之前生成的组合重复了。
- △：当前求和 > target，不能选下去了，返回。
- ○：求和正好 == target，加入解集，并返回。

![](images/39.png)

利用约束条件剪枝

利用后两个约束条件做剪枝，较为简单，设置递归出口如下：

```go
		if target <= 0 {
			if target == 0 { // 找到一组正确组合
				res = append(res, append([]int(nil), comb...)) // 将当前组合加入解集
			}
			return // 结束当前递归
		}
```


**不产生重复组合怎么限制（剪枝）？**

如图，只要限制下一次选择的起点，是基于本次的选择，这样下一次就不会选到本次选择同层左边的数。即通过控制 for 遍历的起点，去掉会产生重复组合的选项。


```go
		for i := index; i < len(candidates); i++ { // 枚举当前可选的数，从index开始
			comb = append(comb, candidates[i]) // 选这个数,基于此，继续选择，传i，下次就不会选到i左边的数
			dfs(target-candidates[i], i)       // 注意这里迭代的时候 index 依旧不变，因为一个元素可以取多次
			comb = comb[:len(comb)-1]          // 撤销选择，回到选择candidates[i]之前的状态，继续尝试选同层右边的数
		}
```

注意，子递归传了 i 而不是 i+1 ，因为元素可以重复选入集合，如果传 i+1 就不重复了。



```go
func combinationSum(candidates []int, target int) [][]int {
	comb, res := []int{}, [][]int{}
	var dfs func(int, int)

	dfs = func(target, index int) {
		if target <= 0 {
			if target == 0 { // 找到一组正确组合
				res = append(res, append([]int(nil), comb...)) // 将当前组合加入解集
			}
			return // 结束当前递归
		}
		for i := index; i < len(candidates); i++ { // 枚举当前可选的数，从index开始
			comb = append(comb, candidates[i]) // 选这个数,基于此，继续选择，传i，下次就不会选到i左边的数
			dfs(target-candidates[i], i)       // 注意这里迭代的时候 index 依旧不变，因为一个元素可以取多次
			comb = comb[:len(comb)-1]          // 撤销选择，回到选择candidates[i]之前的状态，继续尝试选同层右边的数
		}
	}
	dfs(target, 0)
	return res
}
```



```go
func combinationSum(candidates []int, target int) (res [][]int) {
	path := []int{}
	sort.Ints(candidates)
	var dfs func(int, int)

	dfs = func(target, index int) {
		if target <= 0 {
			if target == 0 {
				res = append(res, append([]int(nil), path...))
			}
			return
		}
		for i := index; i < len(candidates); i++ { // 枚举当前可选的数，从index开始
			if candidates[i] > target { // 剪枝优化
				break
			}
			path = append(path, candidates[i]) // 选这个数,基于此，继续选择，传i，下次就不会选到i左边的数
			dfs(target-candidates[i], i)       // 注意这里迭代的时候 index 依旧不变，因为一个元素可以取多次
			path = path[:len(path)-1]          // 撤销选择，回到选择candidates[i]之前的状态，继续尝试选同层右边的数
		}
	}

	dfs(target, 0)
	return
}

```

[参考](https://leetcode.cn/problems/combination-sum/solution/shou-hua-tu-jie-zu-he-zong-he-combination-sum-by-x/)






## [62. 不同路径](https://leetcode-cn.com/problems/unique-paths/)


**方法一：动态规划**

![截屏2021-04-20 10.00.31.png](http://ww1.sinaimg.cn/large/007daNw2ly1gpq0altspuj31hw0t4dk1.jpg)

![截屏2021-04-20 10.01.56.png](http://ww1.sinaimg.cn/large/007daNw2ly1gpq0assvhfj31sc0scqb0.jpg)


```go
func uniquePaths(m int, n int) int {
	dp := make([][]int, m)
	for i := 0; i < m; i++ {
		dp[i] = make([]int, n)
		dp[i][0] = 1
	}
	for j := 0; j < n; j++ {
		dp[0][j] = 1
	}
	for i := 1; i < m; i++ {
		for j := 1; j < n; j++ {
			dp[i][j] = dp[i-1][j] + dp[i][j-1]
		}
	}
	return dp[m-1][n-1]
}
```

复杂度分析

- 时间复杂度：O(mn)。

- 空间复杂度：O(mn)，即为存储所有状态需要的空间。注意到 dp[i][j] 仅与第 i 行和第 i−1 行的状态有关，因此我们可以使用滚动数组代替代码中的二维数组，使空间复杂度降低为 O(n)。此外，由于我们交换行列的值并不会对答案产生影响，因此我们总可以通过交换 m 和 n 使得 m≤n，这样空间复杂度降低至 O(min(m,n))。

**优化**

```go
func uniquePaths(m int, n int) int {
	dp := make([]int, n)
	for j := 0; j < n; j++ {
		dp[j] = 1
	}
	for i := 1; i < m; i++ {
		for j := 1; j < n; j++ {
			dp[j] += dp[j-1]
		}
	}
	return dp[n-1]
}
```
复杂度分析

- 时间复杂度：O(mn)。

- 空间复杂度：O(n)，即为存储所有状态需要的空间。注意到 dp[i][j] 仅与第 i 行和第 i−1 行的状态有关，因此我们可以使用滚动数组代替代码中的二维数组，使空间复杂度降低为 O(n)。此外，由于我们交换行列的值并不会对答案产生影响，因此我们总可以通过交换 m 和 n 使得 m≤n，这样空间复杂度降低至 O(min(m,n))。












