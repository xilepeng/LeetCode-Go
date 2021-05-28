

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

[补充题2. 圆环回原点问题](https://mp.weixin.qq.com/s/VnGFEWHeD3nh1n9JSDkVUg)



------

[79. 单词搜索](https://leetcode-cn.com/problems/word-search/)

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

[9. 回文数](https://leetcode-cn.com/problems/palindrome-number/)



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


[剑指 Offer 10- II. 青蛙跳台阶问题](https://leetcode-cn.com/problems/qing-wa-tiao-tai-jie-wen-ti-lcof/)

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

[剑指 Offer 10- I. 斐波那契数列](https://leetcode-cn.com/problems/fei-bo-na-qi-shu-lie-lcof/)

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




[70. 爬楼梯](https://leetcode-cn.com/problems/climbing-stairs/)

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


[剑指 Offer 62. 圆圈中最后剩下的数字](https://leetcode-cn.com/problems/yuan-quan-zhong-zui-hou-sheng-xia-de-shu-zi-lcof/)



[剑指 Offer 27. 二叉树的镜像](https://leetcode-cn.com/problems/er-cha-shu-de-jing-xiang-lcof/)


[226. 翻转二叉树](https://leetcode-cn.com/problems/invert-binary-tree/)

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
	stack := []*TreeNode{root}
	for len(stack) != 0 {
		node := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		node.Left, node.Right = node.Right, node.Left //交换左右子树
		if node.Left != nil {
			stack = append(stack, node.Left)
		}
		if node.Right != nil {
			stack = append(stack, node.Right)
		}
	}
	return root
}
```












[补充题23. 检测循环依赖](https://mp.weixin.qq.com/s/q6AhBt6MX2RL_HNZc8cYKQ)


```go

```

[207. 课程表](https://leetcode-cn.com/problems/course-schedule/)

```go
func canFinish(numCourses int, prerequisites [][]int) bool {
	T := make([][]int, numCourses)       // 存储有向图
	in_degree := make([]int, numCourses) // 存储每个节点的入度
	res := make([]int, 0, numCourses)    // 存储答案

	for _, v := range prerequisites {
		T[v[1]] = append(T[v[1]], v[0])
		in_degree[v[0]]++
	}

	q := []int{}
	for i := 0; i < numCourses; i++ {
		if in_degree[i] == 0 { // 将所有入度为 0 的节点放入队列中
			q = append(q, i)
		}
	}

	for len(q) > 0 {
		u := q[0] // 从队首取出一个节点
		q = q[1:]
		res = append(res, u) // 放入答案中
		for _, v := range T[u] {
			in_degree[v]--
			if in_degree[v] == 0 { // 如果相邻节点 v 的入度为 0
				q = append(q, v) // 可以选 v 对应的课程了
			}
		}
	}
	return len(res) == numCourses
}
```

[210. 课程表 II](https://leetcode-cn.com/problems/course-schedule-ii/)





[739. 每日温度](https://leetcode-cn.com/problems/daily-temperatures/)

[26. 删除有序数组中的重复项](https://leetcode-cn.com/problems/remove-duplicates-from-sorted-array/)

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

[287. 寻找重复数](https://leetcode-cn.com/problems/find-the-duplicate-number/)

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


[11. 盛最多水的容器](https://leetcode-cn.com/problems/container-with-most-water/)

[560. 和为K的子数组](https://leetcode-cn.com/problems/subarray-sum-equals-k/)

[443. 压缩字符串](https://leetcode-cn.com/problems/string-compression/)

[50. Pow(x, n)](https://leetcode-cn.com/problems/powx-n/)

[补充题2. 圆环回原点问题](https://mp.weixin.qq.com/s/VnGFEWHeD3nh1n9JSDkVUg)




![群二维码地址](http://ww1.sinaimg.cn/large/007daNw2ly1gqvw912ofij60fo0ludit02.jpg)


![过期点我](http://ww1.sinaimg.cn/large/007daNw2ly1gqvm5w0rjvj30fo0lu41c.jpg)
