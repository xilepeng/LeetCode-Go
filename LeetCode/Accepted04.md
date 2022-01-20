[83. 删除排序链表中的重复元素](https://leetcode-cn.com/problems/remove-duplicates-from-sorted-list/)

[112. 路径总和](https://leetcode-cn.com/problems/path-sum/)

[113. 路径总和 II](https://leetcode-cn.com/problems/path-sum-ii/) 补充

[239. 滑动窗口最大值](https://leetcode-cn.com/problems/sliding-window-maximum/)

[165. 比较版本号](https://leetcode-cn.com/problems/compare-version-numbers/)



[93. 复原 IP 地址](https://leetcode-cn.com/problems/restore-ip-addresses/) 

[98. 验证二叉搜索树](https://leetcode-cn.com/problems/validate-binary-search-tree/)

[129. 求根节点到叶节点数字之和](https://leetcode-cn.com/problems/sum-root-to-leaf-numbers/)


[48. 旋转图像](https://leetcode-cn.com/problems/rotate-image/)

[240. 搜索二维矩阵 II](https://leetcode-cn.com/problems/search-a-2d-matrix-ii/)

[78. 子集](https://leetcode-cn.com/problems/subsets/)

[153. 寻找旋转排序数组中的最小值](https://leetcode-cn.com/problems/find-minimum-in-rotated-sorted-array/)

[39. 组合总和](https://leetcode-cn.com/problems/combination-sum/)

[64. 最小路径和](https://leetcode-cn.com/problems/minimum-path-sum/)



[136. 只出现一次的数字](https://leetcode-cn.com/problems/single-number/)

[34. 在排序数组中查找元素的第一个和最后一个位置](https://leetcode-cn.com/problems/find-first-and-last-position-of-element-in-sorted-array/)

[41. 缺失的第一个正数](https://leetcode-cn.com/problems/first-missing-positive/)



[958. 二叉树的完全性检验](https://leetcode-cn.com/problems/check-completeness-of-a-binary-tree/)

[62. 不同路径](https://leetcode-cn.com/problems/unique-paths/)

[34. 在排序数组中查找元素的第一个和最后一个位置](https://leetcode-cn.com/problems/find-first-and-last-position-of-element-in-sorted-array/)

------




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



[112. 路径总和](https://leetcode-cn.com/problems/path-sum/)

### 方法一：递归

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

[113. 路径总和 II](https://leetcode-cn.com/problems/path-sum-ii/) 补充

```go
/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */
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
/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */
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




[239. 滑动窗口最大值](https://leetcode-cn.com/problems/sliding-window-maximum/)

### 方法一 暴力解法 O(nk)

```go
func maxSlidingWindow(nums []int, k int) []int {
	res, n := make([]int, 0, k), len(nums)
	if n == 0 {
		return make([]int, 0)
	}
	for i := 0; i <= n-k; i++ {
		max := nums[i]
		for j := 1; j < k; j++ {
			if max < nums[i+j] {
				max = nums[i+j]
			}
		}
		res = append(res, max)
	}
	return res
}
```
Time Limit Exceeded
50/61 cases passed (N/A)

### 方法二 双端队列 Deque
最优的解法是用双端队列，队列的一边永远都存的是窗口的最大值，队列的另外一个边存的是比最大值小的值。队列中最大值左边的所有值都出队。在保证了双端队列的一边即是最大值以后，时间复杂度是 O(n)，空间复杂度是 O(K)

```go
func maxSlidingWindow(nums []int, k int) []int {
	if len(nums) == 0 || len(nums) < k {
		return make([]int, 0)
	}
	window := make([]int, 0, k) // store the index of nums
	result := make([]int, 0, len(nums)-k+1)
	for i, v := range nums {
		if i >= k && window[0] <= i-k { // if the left-most index is out of window, remove it
			window = window[1:len(window)]
		}
		for len(window) > 0 && nums[window[len(window)-1]] < v {
			window = window[0 : len(window)-1]
		}
		window = append(window, i)
		if i >= k-1 {
			result = append(result, nums[window[0]]) // the left-most is the index of max value in nums
		}
	}
	return result
}
```

![](http://ww1.sinaimg.cn/large/007daNw2ly1gpmcfyuvh1j319g0mun0a.jpg)

```go
func maxSlidingWindow(nums []int, k int) []int {
	q, res := []int{}, []int{}
	for i := 0; i < len(nums); i++ {
		if len(q) > 0 && i-k+1 > q[0] {
			q = q[1:] //窗口满了，删除队头
		}
		for len(q) > 0 && nums[q[len(q)-1]] <= nums[i] {
			q = q[:len(q)-1] //队尾小于当前元素，删除队尾
		}
		q = append(q, i)
		if i >= k-1 { //窗口大小大于等于 k
			res = append(res, nums[q[0]])
		}
	}
	return res
}
```


[165. 比较版本号](https://leetcode-cn.com/problems/compare-version-numbers/)

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










[93. 复原 IP 地址](https://leetcode-cn.com/problems/restore-ip-addresses/)

思路

- 以 "25525511135" 为例，做第一步时我们有几种选择？

1. 选 "2" 作为第一个片段
2. 选 "25" 作为第一个片段
3. 选 "255" 作为第一个片段
- 能切三种不同的长度，切第二个片段时，又面临三种选择。
- 这会向下分支形成一棵树，我们用 DFS 去遍历所有选择，必要时提前回溯。
	因为某一步的选择可能是错的，得不到正确的结果，不要往下做了。撤销最后一个选择，回到选择前的状态，去试另一个选择。
- 回溯的第一个要点：选择，它展开了一颗空间树。

#### 回溯的要点二——约束
- 约束条件限制了当前的选项，这道题的约束条件是：
1. 一个片段的长度是 1~3
2. 片段的值范围是 0~255
3. 不能是 "0x"、"0xx" 形式（测试用例告诉我们的）
- 用这些约束进行充分地剪枝，去掉一些选择，避免搜索「不会产生正确答案」的分支。
#### 回溯的要点三——目标
- 目标决定了什么时候捕获答案，什么时候砍掉死支，回溯。
- 目标是生成 4 个有效片段，并且要耗尽 IP 的字符。
- 当满足该条件时，说明生成了一个有效组合，加入解集，结束当前递归，继续探索别的分支。
- 如果满4个有效片段，但没耗尽字符，不是想要的解，不继续往下递归，提前回溯。
#### 定义 dfs 函数
- dfs 函数传什么？也就是，用什么描述一个节点的状态？
- 选择切出一个片段后，继续递归剩余子串。可以传子串，也可以传指针，加上当前的片段数组，描述节点的状态。
- dfs 函数做的事：复原从 start 到末尾的子串。

我把递归树画了出来，可以看看回溯的细节：


![](https://pic.leetcode-cn.com/5276b1631cb1fc47d8d88dd021f1302213291bf05bfdfdc6209370ce9034be83-image.png)

如图['2','5','5','2']未耗尽字符，不是有效组合，不继续选下去。撤销选择"2"，回到之前的状态（当前分支砍掉了），切入到另一个分支，选择"25"。

回溯会穷举所有节点，通常用于解决「找出所有可能的组合」问题。

下图展示找到一个有效的组合的样子。start 指针越界，代表耗尽了所有字符，且满 4 个片段。


![](https://pic.leetcode-cn.com/e3e3a6dac1ecb79da18740f7968a5eedaa80d5a0e0e45463c7096f663748e0fa-image.png)

```go
func restoreIpAddresses(s string) []string {
	res := []string{}
	var dfs func([]string, int)

	dfs = func(sub []string, start int) {
		if len(sub) == 4 && start == len(s) { // 片段满4段，且耗尽所有字符
			res = append(res, strings.Join(sub, ".")) // 拼成字符串，加入解集
			return
		}
		if len(sub) == 4 && start < len(s) { // 满4段，字符未耗尽，不用往下选了
			return
		}
		for length := 1; length <= 3; length++ { // 枚举出选择，三种切割长度
			if start+length-1 >= len(s) { // 加上要切的长度就越界，不能切这个长度
				return
			}
			if length != 1 && s[start] == '0' { // 不能切出'0x'、'0xx'
				return
			}
			str := s[start : start+length]          // 当前选择切出的片段
			if n, _ := strconv.Atoi(str); n > 255 { // 不能超过255
				return
			}
			sub = append(sub, str) // 作出选择，将片段加入sub
			dfs(sub, start+length) // 基于当前选择，继续选择，注意更新指针
			sub = sub[:len(sub)-1] // 上面一句的递归分支结束，撤销最后的选择，进入下一轮迭代，考察下一个切割长度
		}
	}
	dfs([]string{}, 0)
	return res
}
```




[98. 验证二叉搜索树](https://leetcode-cn.com/problems/validate-binary-search-tree/)

### 方法一: 递归
思路和算法

![](https://assets.leetcode-cn.com/solution-static/98/1.PNG)
![](https://assets.leetcode-cn.com/solution-static/98/2.PNG)
![](https://assets.leetcode-cn.com/solution-static/98/3.PNG)
![](https://assets.leetcode-cn.com/solution-static/98/4.PNG)

解法一，直接按照定义比较大小，比 root 节点小的都在左边，比 root 节点大的都在右边
```go
func isValidBST(root *TreeNode) bool {
	return isValidbst(root, -1<<63, 1<<63-1)
}
func isValidbst(root *TreeNode, min, max int) bool {
	if root == nil {
		return true
	}
	v := root.Val
	return min < v && v < max && isValidbst(root.Left, min, v) && isValidbst(root.Right, v, max)
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


```go
func isValidBST(root *TreeNode) bool {
	return dfs(root, math.MinInt64, math.MaxInt64)
}
func dfs(root *TreeNode, lower, upper int) bool {
	if root == nil {
		return true
	}
	if root.Val <= lower || root.Val >= upper {
		return false
	}
	return dfs(root.Left, lower, root.Val) && dfs(root.Right, root.Val, upper)
}
```

```go
func isValidBST(root *TreeNode) bool {
	var dfs func(*TreeNode, int, int) bool
	dfs = func(root *TreeNode, lower, upper int) bool {
		if root == nil {
			return true
		}
		if root.Val <= lower || root.Val >= upper {
			return false
		}
		return dfs(root.Left, lower, root.Val) && dfs(root.Right, root.Val, upper)
	}
	return dfs(root, math.MinInt64, math.MaxInt64)
}
```
复杂度分析

时间复杂度 : O(n)，其中 n 为二叉树的节点个数。在递归调用的时候二叉树的每个节点最多被访问一次，因此时间复杂度为 O(n)。

空间复杂度 : O(n)，其中 n 为二叉树的节点个数。递归函数在递归过程中需要为每一层递归函数分配栈空间，所以这里需要额外的空间且该空间取决于递归的深度，即二叉树的高度。最坏情况下二叉树为一条链，树的高度为 n ，递归最深达到 n 层，故最坏情况下空间复杂度为 O(n) 。




### 方法二：中序遍历
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




[129. 求根节点到叶节点数字之和](https://leetcode-cn.com/problems/sum-root-to-leaf-numbers/)

### 方法一：深度优先搜索
思路与算法

从根节点开始，遍历每个节点，如果遇到叶子节点，则将叶子节点对应的数字加到数字之和。如果当前节点不是叶子节点，则计算其子节点对应的数字，然后对子节点递归遍历。

![](https://assets.leetcode-cn.com/solution-static/129/fig1.png)

![](https://pic.leetcode-cn.com/1603933660-UNWQbT-image.png)

```go
/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */
func sumNumbers(root *TreeNode) int {
	var dfs func(*TreeNode, int) int

	dfs = func(root *TreeNode, prevSum int) int {
		if root == nil {
			return 0
		}
		sum := prevSum*10 + root.Val
		if root.Left == nil && root.Right == nil {
			return sum
		}
		return dfs(root.Left, sum) + dfs(root.Right, sum)
	}

	return dfs(root, 0)
}
```

复杂度分析

- 时间复杂度：O(n)，其中 n 是二叉树的节点个数。对每个节点访问一次。

- 空间复杂度：O(n)，其中 n 是二叉树的节点个数。空间复杂度主要取决于递归调用的栈空间，递归栈的深度等于二叉树的高度，最坏情况下，二叉树的高度等于节点个数，空间复杂度为 O(n)。






[78. 子集](https://leetcode-cn.com/problems/subsets/)


### 方法一：位运算

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

### 方法一：迭代法实现子集枚举
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



### 方法二：递归法实现子集枚举

#### 思路 1
- 单看每个元素，都有两种选择：选入子集，或不选入子集。
- 比如[1,2,3]，先看1，选1或不选1，都会再看2，选2或不选2，以此类推。
- 考察当前枚举的数，基于选它而继续，是一个递归分支；基于不选它而继续，又是一个分支。

![1.png](http://ww1.sinaimg.cn/large/007daNw2ly1gpms453jltj31fw0jjq6r.jpg)

- 用索引index代表当前递归考察的数字nums[index]。
- 当index越界时，所有数字考察完，得到一个解，位于递归树的底部，把它加入解集，结束当前递归分支。

#### 为什么要回溯？
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

#### 思路2
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

[48. 旋转图像](https://leetcode-cn.com/problems/rotate-image/)

### 方法一：用翻转代替旋转

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




[240. 搜索二维矩阵 II](https://leetcode-cn.com/problems/search-a-2d-matrix-ii/)
![](https://pic.leetcode-cn.com/Figures/240/Slide3.PNG)

## 方法一：模拟 

1. 从右上角开始搜索

```go
func searchMatrix(matrix [][]int, target int) bool {
	row, col := 0, len(matrix[0])-1
	for row <= len(matrix)-1 && col >= 0 {
		if target == matrix[row][col] {
			return true
		} else if target < matrix[row][col] {
			col--
		} else {
			row++
		}
	}
	return false
}
```

2. 从左下角开始搜索

```go
func searchMatrix(matrix [][]int, target int) bool {
	row, col := len(matrix)-1, 0
	for row >= 0 && col <= len(matrix[0])-1 {
		if target == matrix[row][col] {
			return true
		} else if target < matrix[row][col] {
			row--
		} else {
			col++
		}
	}
	return false
}
```
复杂度分析

- 时间复杂度：O(n+m)。
时间复杂度分析的关键是注意到在每次迭代（我们不返回 true）时，行或列都会精确地递减/递增一次。由于行只能减少 m 次，而列只能增加 n 次，因此在导致 for 循环终止之前，循环不能运行超过 n+m 次。因为所有其他的工作都是常数，所以总的时间复杂度在矩阵维数之和中是线性的。
- 空间复杂度：O(1)，因为这种方法只处理几个指针，所以它的内存占用是恒定的。

## 方法二：二分法搜索

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




[136. 只出现一次的数字](https://leetcode-cn.com/problems/single-number/)

### 方法一：位运算 

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







[153. 寻找旋转排序数组中的最小值](https://leetcode-cn.com/problems/find-minimum-in-rotated-sorted-array/)

### 思路
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



[34. 在排序数组中查找元素的第一个和最后一个位置](https://leetcode-cn.com/problems/find-first-and-last-position-of-element-in-sorted-array/)

### 方法一：二分查找

### 解题思路 
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
### 方法二：二分查找

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




[39. 组合总和](https://leetcode-cn.com/problems/combination-sum/)

![1.png](http://ww1.sinaimg.cn/large/007daNw2ly1gplqpjg5wdj31ax0jnn10.jpg)

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






[64. 最小路径和](https://leetcode-cn.com/problems/minimum-path-sum/)

### 方法一：动态规划

```go
func minPathSum(grid [][]int) int {
	m, n := len(grid), len(grid[0])
	for i := 1; i < m; i++ {
		grid[i][0] += grid[i-1][0]
	}
	for j := 1; j < n; j++ {
		grid[0][j] += grid[0][j-1]
	}
	for i := 1; i < m; i++ {
		for j := 1; j < n; j++ {
			grid[i][j] += min(grid[i][j-1], grid[i-1][j])
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












[41. 缺失的第一个正数](https://leetcode-cn.com/problems/first-missing-positive/) 


## 方法一：置换
除了打标记以外，我们还可以使用置换的方法，将给定的数组「恢复」成下面的形式：

如果数组中包含 x x∈[1,N]，那么恢复后，数组的第 x - 1 个元素为 x。

```go
func firstMissingPositive(nums []int) int {
	n := len(nums)
	for i := 0; i < n; i++ {
		for nums[i] > 0 && nums[i] <= n && nums[nums[i]-1] != nums[i] {
			nums[nums[i]-1], nums[i] = nums[i], nums[nums[i]-1]
		}
	}
	for i := 0; i < n; i++ {
		if nums[i] != i+1 {
			return i + 1
		}
	}
	return n + 1
}
```


复杂度分析

- 时间复杂度：O(N)，其中 N 是数组的长度。

- 空间复杂度：O(1)。



## 方法二：哈希表

![](https://assets.leetcode-cn.com/solution-static/41/41_fig1.png)


```go
func firstMissingPositive(nums []int) int {
	n := len(nums)
	for i := 0; i < n; i++ { //将小于等于0的数变为 n+1
		if nums[i] <= 0 {
			nums[i] = n + 1
		}
	}
	for _, v := range nums { //将小于等于 n 的元素对应位置变为负数
		num := abs(v)
		if num <= n {
			nums[num-1] = -abs(nums[num-1])
		}
	}
	for i := 0; i < n; i++ {
		if nums[i] > 0 { //返回第一个大于0的元素下标 +1
			return i + 1
		}
	}
	return n + 1
}
func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}
```

复杂度分析

- 时间复杂度：O(N)，其中 N 是数组的长度。

- 空间复杂度：O(1)。



```go
func firstMissingPositive(nums []int) int {
	n := len(nums)
	hash := make(map[int]int, n)
	for _, v := range nums {
		hash[v] = v
	}
	for i := 1; i < n+1; i++ {
		if _, ok := hash[i]; !ok {
			return i
		}
	}
	return n + 1
}
```
复杂度分析

- 时间复杂度：O(N)，其中 N 是数组的长度。

- 空间复杂度：O(N)。












[62. 不同路径](https://leetcode-cn.com/problems/unique-paths/)


### 方法一：动态规划

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

### 优化

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







