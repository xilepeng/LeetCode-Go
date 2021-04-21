
[94. 二叉树的中序遍历](https://leetcode-cn.com/problems/binary-tree-inorder-traversal/)

[46. 全排列](https://leetcode-cn.com/problems/permutations/)

[47. 全排列 II](https://leetcode-cn.com/problems/permutations-ii/)  补充

[200. 岛屿数量](https://leetcode-cn.com/problems/number-of-islands/)

[5. 最长回文子串](https://leetcode-cn.com/problems/longest-palindromic-substring/)

[232. 用栈实现队列](https://leetcode-cn.com/problems/implement-queue-using-stacks/)

[54. 螺旋矩阵](https://leetcode-cn.com/problems/spiral-matrix/)

[33. 搜索旋转排序数组](https://leetcode-cn.com/problems/search-in-rotated-sorted-array/)

[92. 反转链表 II](https://leetcode-cn.com/problems/reverse-linked-list-ii/)

[199. 二叉树的右视图](https://leetcode-cn.com/problems/binary-tree-right-side-view/)

[300. 最长递增子序列](https://leetcode-cn.com/problems/longest-increasing-subsequence/)


------


[94. 二叉树的中序遍历](https://leetcode-cn.com/problems/binary-tree-inorder-traversal/)

```go
func inorderTraversal(root *TreeNode) (res []int) {
	var inorder func(node *TreeNode)
	inorder = func(node *TreeNode) {
		if node == nil {
			return
		}
		inorder(node.Left)
		res = append(res, node.Val)
		inorder(node.Right)
	}
	inorder(root)
	return
}
```

```go
func inorderTraversal(root *TreeNode) (res []int) {
	stack := []*TreeNode{}
	for root != nil || len(stack) > 0 {
		for root != nil {
			stack = append(stack, root)
			root = root.Left
		}
		root = stack[len(stack)-1]   //取栈顶
		stack = stack[:len(stack)-1] //出栈
		res = append(res, root.Val)
		root = root.Right
	}
	return
}
```



[46. 全排列](https://leetcode-cn.com/problems/permutations/)


### 方法一：枚举每个位置放哪个数 (回溯)

![截屏2021-04-04 19.05.15.png](http://ww1.sinaimg.cn/large/007daNw2ly1gp7x3wg42ij30zk0kwtb0.jpg)

我们从前往后，一位一位枚举，每次选择一个没有被使用过的数。
选好之后，将该数的状态改成“已被使用”，同时将该数记录在相应位置上，然后递归。
递归返回时，不要忘记将该数的状态改成“未被使用”，并将该数从相应位置上删除。

#### 闭包实现：

```go
func permute(nums []int) (res [][]int) {
	used, path, n := make([]bool, len(nums)), []int{}, len(nums)
	var dfs func(int)
	dfs = func(pos int) {
		if pos == n {
			res = append(res, append([]int{}, path...))
			return
		}
		for i := range nums {
			if !used[i] {
				used[i] = true
				path = append(path, nums[i])
				dfs(pos + 1)
				used[i] = false
				path = path[:len(path)-1]
			}
		}
	}
	dfs(0)
	return
}
```

![截屏2021-04-04 12.47.06.png](http://ww1.sinaimg.cn/large/007daNw2ly1gp7m6z5viej30tw0ly0v8.jpg)

```go
func permute(nums []int) [][]int {
	res, n := [][]int{}, len(nums)
	var dfs func(int)
	dfs = func(pos int) {
		if pos == len(nums) {
			res = append(res, append([]int{}, nums...))
		}
		for i := pos; i < n; i++ {
			nums[i], nums[pos] = nums[pos], nums[i]
			dfs(pos + 1)
			nums[i], nums[pos] = nums[pos], nums[i]
		}
	}
	dfs(0)
	return res
}
```


复杂度分析

- 时间复杂度：O(n×n!)，其中 nn 为序列的长度。
- 空间复杂度：O(n)，其中 n 为序列的长度。除答案数组以外，递归函数在递归过程中需要为每一层递归函数分配栈空间，所以这里需要额外的空间且该空间取决于递归的深度，这里可知递归调用深度为 O(n)。

#### 为什么加入解集时，要将数组（在go中是切片）内容拷贝到一个新的数组里，再加入解集。

这个 path 变量是一个地址引用，结束当前递归，将它加入 res，后续的递归分支还要继续进行搜索，还要继续传递这个 path，这个地址引用所指向的内存空间还要继续被操作，所以 res 中的 path 所引用的内容会被改变，这就不对，所以要拷贝一份内容，到一份新的数组里，然后放入 res，这样后续对 path 的操作，就不会影响已经放入 res 的内容。


[47. 全排列 II](https://leetcode-cn.com/problems/permutations-ii/)

### 方法一：枚举每个位置放哪个数 (回溯)
![截屏2021-04-04 19.05.15.png](http://ww1.sinaimg.cn/large/007daNw2ly1gp7x3wg42ij30zk0kwtb0.jpg)

假设我们有 3 个重复数排完序后相邻，那么我们一定保证每次都是拿从左往右第一个未被填过的数字，即整个数组的状态其实是保证了 
[未填入，未填入，未填入] 到 [填入，未填入，未填入]，再到 [填入，填入，未填入]，最后到 [填入，填入，填入] 的过程的，因此可以达到去重的目标。


dfs 闭包实现：

```go
func permuteUnique(nums []int) (res [][]int) {
	sort.Ints(nums)
	used, path, n := make([]bool, len(nums)), []int{}, len(nums)
	var dfs func(int)
	dfs = func(pos int) {
		if pos == n {
			res = append(res, append([]int{}, path...))
			return
		}
		for i := 0; i < n; i++ {
			if used[i] || i > 0 && !used[i-1] && nums[i-1] == nums[i] {
				continue
			}
			used[i] = true
			path = append(path, nums[i])
			dfs(pos + 1)
			used[i] = false
			path = path[:len(path)-1]	
		}
	}
	dfs(0)
	return
}
```



[200. 岛屿数量](https://leetcode-cn.com/problems/number-of-islands/)

### 思路一：深度优先遍历DFS

- 目标是找到矩阵中 “岛屿的数量” ，上下左右相连的 1 都被认为是连续岛屿。
- dfs方法： 设目前指针指向一个岛屿中的某一点 (i, j)，寻找包括此点的岛屿边界。
	1. 从 (i, j) 向此点的上下左右 (i+1,j),(i-1,j),(i,j+1),(i,j-1) 做深度搜索。
	2. 终止条件：
		- (i, j) 越过矩阵边界;
		- grid[i][j] == 0，代表此分支已越过岛屿边界。
	3. 搜索岛屿的同时，执行 grid[i][j] = '0'，即将岛屿所有节点删除，以免之后重复搜索相同岛屿。

#### 主循环：

遍历整个矩阵，当遇到 grid[i][j] == '1' 时，从此点开始做深度优先搜索 dfs，岛屿数 count + 1 且在深度优先搜索中删除此岛屿。

- 最终返回岛屿数 count 即可。


```go
func numIslands(grid [][]byte) int {
	count := 0
	for i := 0; i < len(grid); i++ {
		for j := 0; j < len(grid[0]); j++ {
			if grid[i][j] == '1' {
				dfs(grid, i, j)
				count++
			}
		}
	}
	return count
}
func dfs(grid [][]byte, i, j int) {
	if (i < 0 || j < 0) || i >= len(grid) || j >= len(grid[0]) || grid[i][j] == '0' {
		return
	}
	grid[i][j] = '0'
	dfs(grid, i+1, j)
	dfs(grid, i-1, j)
	dfs(grid, i, j-1)
	dfs(grid, i, j+1)
}
```
#### 闭包

```go
func numIslands(grid [][]byte) int {
	var dfs func(grid [][]byte, i, j int)
	dfs = func(grid [][]byte, i, j int) {
		if (i < 0 || j < 0) || i >= len(grid) || j >= len(grid[0]) || grid[i][j] == '0' {
			return
		}
		grid[i][j] = '0'
		dfs(grid, i+1, j)
		dfs(grid, i-1, j)
		dfs(grid, i, j-1)
		dfs(grid, i, j+1)
	}
	count := 0
	for i := 0; i < len(grid); i++ {
		for j := 0; j < len(grid[0]); j++ {
			if grid[i][j] == '1' {
				dfs(grid, i, j)
				count++
			}
		}
	}
	return count
}
```


[5. 最长回文子串](https://leetcode-cn.com/problems/longest-palindromic-substring/)

### 方法一：（暴力枚举） O(n^2)

由于字符串长度小于1000，因此我们可以用 O(n^2)的算法枚举所有可能的情况。
首先枚举回文串的中心 i，然后分两种情况向两边扩展边界，直到遇到不同字符为止:

回文串长度是奇数，则依次判断 s[i−k]==s[i+k],k=1,2,3,…
回文串长度是偶数，则依次判断 s[i−k]==s[i+k−1],k=1,2,3,…
如果遇到不同字符，则我们就找到了以 i 为中心的回文串边界。

时间复杂度分析：一共两重循环，所以时间复杂度是 O(n^2)


```go
func longestPalindrome(s string) string {
	res := ""
	for i := 0; i < len(s); i++ {
		for l, r := i, i; l >= 0 && r < len(s) && s[l] == s[r]; l, r = l-1, r+1 {
			if len(res) < r-l+1 {
				res = s[l : r+1]
			}
		}
		for l, r := i, i+1; l >= 0 && r < len(s) && s[l] == s[r]; l, r = l-1, r+1 {
			if len(res) < r-l+1 {
				res = s[l : r+1]
			}
		}
	}
	return res
}
```


### 方法二：中心扩展算法

![截屏2021-04-02 11.43.48.png](http://ww1.sinaimg.cn/large/007daNw2ly1gp59403adbj31kc0i0762.jpg)

```go
func longestPalindrome(s string) string {
	low, maxLen := 0, 0
	for i := range s {
		expand(s, i, i, &low, &maxLen)
		expand(s, i, i+1, &low, &maxLen)
	}
	return s[low : low+maxLen]
}
func expand(s string, l, r int, low, maxLen *int) {
	for ; l >= 0 && r < len(s) && s[l] == s[r]; l, r = l-1, r+1 {
	}
	if *maxLen < r-l-1 {
		*low = l + 1
		*maxLen = r - l - 1
	}
}
```
复杂度分析

- 时间复杂度：O(n^2)，其中 n 是字符串的长度。长度为 1 和 2 的回文中心分别有 n 和 n−1 个，每个回文中心最多会向外扩展 O(n) 次。
- 空间复杂度：O(1)。




```go
func longestPalindrome(s string) string {
	start, end := 0, 0
	for i := range s {
		l, r := expand(s, i, i)
		if end-start < r-l {
			start, end = l, r
		}
		l, r = expand(s, i, i+1)
		if end-start < r-l {
			start, end = l, r
		}
	}
	return s[start : end+1]
}
func expand(s string, l, r int) (int, int) {
	for ; l >= 0 && r < len(s) && s[l] == s[r]; l, r = l-1, r+1 {
	}
	return l + 1, r - 1
}
```





[232. 用栈实现队列](https://leetcode-cn.com/problems/implement-queue-using-stacks/)

### 方法一：双栈


![截屏2021-03-28 20.58.04.png](http://ww1.sinaimg.cn/large/007daNw2ly1gozx18b8ilj31740jsdhb.jpg)

思路

将一个栈当作输入栈，用于压入 push 传入的数据；另一个栈当作输出栈，用于 pop 和 peek 操作。

每次 pop 或 peek 时，若输出栈为空则将输入栈的全部数据依次弹出并压入输出栈，这样输出栈从栈顶往栈底的顺序就是队列从队首往队尾的顺序。



```go
type MyQueue struct {
	inStack, outStack []int
}

/** Initialize your data structure here. */
func Constructor() MyQueue {
	return MyQueue{}
}

/** Push element x to the back of queue. */
func (q *MyQueue) Push(x int) {
	q.inStack = append(q.inStack, x)
}

func (q *MyQueue) in2out() {
	for len(q.inStack) > 0 {
		q.outStack = append(q.outStack, q.inStack[len(q.inStack)-1])
		q.inStack = q.inStack[:len(q.inStack)-1]
	}
}

/** Removes the element from in front of queue and returns that element. */
func (q *MyQueue) Pop() int {
	if len(q.outStack) == 0 {
		q.in2out()
	}
	x := q.outStack[len(q.outStack)-1]
	q.outStack = q.outStack[:len(q.outStack)-1]
	return x
}

/** Get the front element. */
func (q *MyQueue) Peek() int {
	if len(q.outStack) == 0 {
		q.in2out()
	}
	return q.outStack[len(q.outStack)-1]
}

/** Returns whether the queue is empty. */
func (q *MyQueue) Empty() bool {
	return len(q.inStack) == 0 && len(q.outStack) == 0
}
```

复杂度分析

- 时间复杂度：push 和 empty 为 O(1)，pop 和 peek 为均摊 O(1)。对于每个元素，至多入栈和出栈各两次，故均摊复杂度为 O(1)。

- 空间复杂度：O(n)。其中 n 是操作总数。对于有 n 次 push 操作的情况，队列中会有 n 个元素，故空间复杂度为 O(n)。



[54. 螺旋矩阵](https://leetcode-cn.com/problems/spiral-matrix/)

![1.png](http://ww1.sinaimg.cn/large/007daNw2ly1goyc6zkdssj31ef0eiwjn.jpg)

- 如果一条边从头遍历到底，则下一条边遍历的起点随之变化

- 选择不遍历到底，可以减小横向、竖向遍历之间的影响

- 轮迭代结束时，4条边的两端同时收窄 1

- 轮迭代所做的事情很清晰：遍历一个“圈”，遍历的范围收缩为内圈

- 层层向里处理，按顺时针依次遍历：上、右、下、左。

- 不再形成“环”了，就会剩下一行或一列，然后单独判断

### 四个边界

- 上边界 top : 0
- 下边界 bottom : matrix.length - 1
- 左边界 left : 0
- 右边界 right : matrix[0].length - 1

### 矩阵不一定是方阵
- top < bottom && left < right 是循环的条件
- 无法构成“环”了，就退出循环，退出时可能是这 3 种情况之一：
- top == bottom && left < right —— 剩一行
- top < bottom && left == right —— 剩一列
- top == bottom && left == right —— 剩一项（也是一行/列）

### 处理剩下的单行或单列
- 因为是按顺时针推入结果数组的，所以
- 剩下的一行，从左至右 依次推入结果数组
- 剩下的一列，从上至下 依次推入结果数组

### 代码

每个元素访问一次，时间复杂度 O(m*n)，m、n 分别是矩阵的行数和列数
空间复杂度 O(m*n)

```go
func spiralOrder(matrix [][]int) []int {
    if len(matrix) == 0 {
        return []int{}
    }
    res := []int{}
    top, bottom, left, right := 0, len(matrix)-1, 0, len(matrix[0])-1
    
    for top < bottom && left < right {
        for i := left; i < right; i++ { res = append(res, matrix[top][i]) }
        for i := top; i < bottom; i++ { res = append(res, matrix[i][right]) }
        for i := right; i > left; i-- { res = append(res, matrix[bottom][i]) }
        for i := bottom; i > top; i-- { res = append(res, matrix[i][left]) }
        right--
        top++
        bottom--
        left++ 
    }
    if top == bottom {
        for i := left; i <= right; i++ { res = append(res, matrix[top][i]) }
    } else if left == right {
        for i := top; i <= bottom; i++ { res = append(res, matrix[i][left]) }
    } 
    return res
}
```

## 换一种遍历的策略：遍历到底

![2.png](http://ww1.sinaimg.cn/large/007daNw2ly1goyc7ez0y4j31l00fswk5.jpg)


- 循环的条件改为： top <= bottom && left <= right
- 每遍历一条边，下一条边遍历的起点被“挤占”，要更新相应的边界
- 值得注意的是，可能出现 在循环中途，不再满足循环的条件 ，即出现 top > bottom || left > right ，其中一对边界彼此交错了
- 这意味着此时所有项都遍历完了，如果没有及时 break ，就会重复遍历

### 解决办法

- 每遍历完一条边，更新完相应的边界后，都加一条判断 if (top > bottom || left > right) break，避免遍历完成时没有及时退出，导致重复遍历。
- 但你发现，遍历完成要么发生在遍历完“上边”，要么发生在遍历完“右边”
- 所以只需在这两步操作之后，加 if (top > bottom || left > right) break 即可

```go
func spiralOrder(matrix [][]int) []int {
	if len(matrix) == 0 {
		return []int{}
	}
	res := []int{}
	top, bottom, left, right := 0, len(matrix)-1, 0, len(matrix[0])-1
	for top <= bottom && left <= right {
		for i := left; i <= right; i++ { res = append(res, matrix[top][i]) }
		top++ 
		for i := top; i <= bottom; i++ { res = append(res, matrix[i][right]) }
		right--
		if top > bottom || left > right { break}
		for i := right; i >= left; i-- { res = append(res, matrix[bottom][i]) }
        bottom--
		for i := bottom; i >= top; i-- { res = append(res, matrix[i][left]) }
        left++
	}
	return res
}
```

### 换一种循环的条件，也是可以的
- 遍历完所有项时，res 数组构建完毕。我们可以用 res 数组的长度 等于 矩阵的项的个数，作为循环的结束条件
- 不等于就继续遍历，等于就 break

```go
func spiralOrder(matrix [][]int) []int {
    if len(matrix) == 0 {
        return []int{}
    }
    res := []int{}
    top, bottom, left, right := 0, len(matrix)-1, 0, len(matrix[0])-1
    size := len(matrix) * len(matrix[0])
    
    for len(res) != size {
        for i := left; i <= right; i++ { res = append(res, matrix[top][i]) }
        top++
        for i := top; i <= bottom; i++ { res = append(res, matrix[i][right]) }
        right--
        if len(res) == size { break } 
        for i := right; i >= left; i-- { res = append(res, matrix[bottom][i]) }
        bottom--
        for i := bottom; i >= top; i-- { res = append(res, matrix[i][left]) }
        left++ 
    }

    return res
}

```



[33. 搜索旋转排序数组](https://leetcode-cn.com/problems/search-in-rotated-sorted-array/)

![](https://assets.leetcode-cn.com/solution-static/33/33_fig1.png)

```go
func search(nums []int, target int) int {
	if len(nums) == 0 {
		return -1
	}
	l, r := 0, len(nums)-1
	for l <= r {
		mid := (l + r) >> 1
		if nums[mid] == target {
			return mid
		}
		if nums[l] <= nums[mid] { //左边有序
			if nums[l] <= target && target < nums[mid] {
				r = mid - 1
			} else {
				l = mid + 1
			}
		} else {
			if nums[mid] < target && target <= nums[r] {
				l = mid + 1
			} else {
				r = mid - 1
			}
		}
	}
	return -1
}
```





[92. 反转链表 II](https://leetcode-cn.com/problems/reverse-linked-list-ii/)

### 方法一：双指针 

```go
/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */

func reverseBetween(head *ListNode, left int, right int) *ListNode {
	dummy := &ListNode{Next: head}
	prevLeft := dummy
	for i := 0; i < left-1; i++ {
		prevLeft = prevLeft.Next
	}
	prev := prevLeft.Next
	curr := prev.Next
	for i := 0; i < right-left; i++ {
		next := curr.Next
		curr.Next = prev
		prev = curr
		curr = next
	}
	prevLeft.Next.Next = curr
	prevLeft.Next = prev
	return dummy.Next
}
```



### 方法二：头插法 

![](https://pic.leetcode-cn.com/1615105232-cvTINs-image.png)

整体思想是：在需要反转的区间里，每遍历到一个节点，让这个新节点来到反转部分的起始位置。下面的图展示了整个流程。

![](https://pic.leetcode-cn.com/1615105242-ZHlvOn-image.png)

下面我们具体解释如何实现。使用三个指针变量 pre、curr、next 来记录反转的过程中需要的变量，它们的意义如下：

- curr：指向待反转区域的第一个节点 left；
- next：永远指向 curr 的下一个节点，循环过程中，curr 变化以后 next 会变化；
- pre：永远指向待反转区域的第一个节点 left 的前一个节点，在循环过程中不变。
第 1 步，我们使用 ①、②、③ 标注「穿针引线」的步骤。

![](https://pic.leetcode-cn.com/1615105296-bmiPxl-image.png)

操作步骤：

- 先将 curr 的下一个节点记录为 next；
- 执行操作 ①：把 curr 的下一个节点指向 next 的下一个节点；
- 执行操作 ②：把 next 的下一个节点指向 pre 的下一个节点；
- 执行操作 ③：把 pre 的下一个节点指向 next。
第 1 步完成以后「拉直」的效果如下：

![](https://pic.leetcode-cn.com/1615105340-UBnTBZ-image.png)

第 2 步，同理。同样需要注意 「穿针引线」操作的先后顺序。

![](https://pic.leetcode-cn.com/1615105353-PsCmzb-image.png)

第 2 步完成以后「拉直」的效果如下：

![](https://pic.leetcode-cn.com/1615105364-aDIFqy-image.png)

第 3 步，同理。
![](https://pic.leetcode-cn.com/1615105376-jIyGwv-image.png)

第 3 步完成以后「拉直」的效果如下：
![](https://pic.leetcode-cn.com/1615105395-EJQnMe-image.png)




```go
func reverseBetween(head *ListNode, left int, right int) *ListNode {
	dummy := &ListNode{Next: head}
	prev := dummy
	for i := 0; i < left-1; i++ {
		prev = prev.Next
	}
	curr := prev.Next
	for i := 0; i < right-left; i++ {
		next := curr.Next
		curr.Next = next.Next
		next.Next = prev.Next
		prev.Next = next
	}
	return dummy.Next
}
```






[199. 二叉树的右视图](https://leetcode-cn.com/problems/binary-tree-right-side-view/)

方法一：BFS

思路： 利用 BFS 进行层次遍历，记录下每层的最后一个元素。

时间复杂度： O(N)，每个节点都入队出队了 1 次。

空间复杂度： O(N)，使用了额外的队列空间。

```go
func rightSideView(root *TreeNode) []int {
	res := []int{}
	if root == nil {
		return res
	}
	queue := []*TreeNode{root}
	for len(queue) != 0 {
		size := len(queue)
		for i := 0; i < size; i++ {
			node := queue[0]
			queue = queue[1:]
			if node.Left != nil {
				queue = append(queue, node.Left)
			}
			if node.Right != nil {
				queue = append(queue, node.Right)
			}
			if i == size-1 {
				res = append(res, node.Val)
			}
		}
	}
	return res
}
```


方法二：DFS 

思路： 我们按照 「根结点 -> 右子树 -> 左子树」 的顺序访问，就可以保证每层都是最先访问最右边的节点的。

（与先序遍历 「根结点 -> 左子树 -> 右子树」 正好相反，先序遍历每层最先访问的是最左边的节点）

时间复杂度： O(N)，每个节点都访问了 1 次。

空间复杂度： O(N)，因为这不是一棵平衡二叉树，二叉树的深度最少是 logN, 最坏的情况下会退化成一条链表，深度就是 N，因此递归时使用的栈空间是 O(N) 的。

```go
var res []int

func rightSideView(root *TreeNode) []int {
	res = []int{}
	dfs(root, 0)
	return res
}
func dfs(root *TreeNode, depth int) {
	if root == nil {
		return
	}
	if depth == len(res) {
		res = append(res, root.Val)
	}
	depth++
	dfs(root.Right, depth)
	dfs(root.Left, depth)
}
```

- 从根节点开始访问，根节点深度是0
- 先访问 当前节点，再递归地访问 右子树 和 左子树。
- 如果当前节点所在深度还没有出现在res里，说明在该深度下当前节点是第一个被访问的节点，因此将当前节点加入res中。











[300. 最长递增子序列](https://leetcode-cn.com/problems/longest-increasing-subsequence/)

### 方法一： nlogn 动态规划 

```go
func lengthOfLIS(nums []int) int {
	dp := []int{}
	for _, num := range nums {
		i := sort.SearchInts(dp, num) //min_index
		if i == len(dp) {
			dp = append(dp, num)
		} else {
			dp[i] = num
		}
	}
	return len(dp)
}
```
复杂度分析

- 时间复杂度：O(nlogn)。数组 nums 的长度为 n，我们依次用数组中的元素去更新 dp 数组，而更新 dp 数组时需要进行 O(logn) 的二分搜索，所以总时间复杂度为 O(nlogn)。

- 空间复杂度：O(n)，需要额外使用长度为 n 的 dp 数组。

#### func SearchInts
- func SearchInts(a []int, x int) int
- SearchInts在递增顺序的a中搜索x，返回x的索引。如果查找不到，返回值是x应该插入a的位置（以保证a的递增顺序），返回值可以是len(a)。

### 方法二：贪心 + 二分查找

![贪心 + 二分查找](http://ww1.sinaimg.cn/large/007daNw2ly1gpbeyy69rdj31ag0lmtb8.jpg)

```go
func lengthOfLIS(nums []int) int {
	d := []int{}
	for _, num := range nums {
		if len(d) == 0 || d[len(d)-1] < num {
			d = append(d, num)
		} else { //二分查找
			l, r := 0, len(d)-1
			pos := r
			for l <= r {
				mid := (l + r) >> 1
				if d[mid] >= num {
					pos = mid
					r = mid - 1
				} else {
					l = mid + 1
				}
			}
			d[pos] = num
		}
	}
	return len(d)
}
```
复杂度分析

- 时间复杂度：O(nlogn)。数组 nums 的长度为 n，我们依次用数组中的元素去更新 d 数组，而更新 d 数组时需要进行 O(logn) 的二分搜索，所以总时间复杂度为 O(nlogn)。

- 空间复杂度：O(n)，需要额外使用长度为 n 的 d 数组。

### 方法三：动态规划

```go
func lengthOfLIS(nums []int) int {
	n := len(nums)
	if n == 0 {
		return 0
	}
	dp, res := make([]int, n), 0
	for i := 0; i < n; i++ {
		dp[i] = 1
		for j := 0; j < i; j++ {
			if nums[j] < nums[i] {
				dp[i] = max(dp[i], dp[j]+1)
			}
		}
		res = max(res, dp[i])
	}
	return res
}
func max(x, y int) int {
	if x > y {
		return x
	}
	return y
}
```
复杂度分析：

- 时间复杂度 O(N^2)： 遍历计算 dp 列表需 O(N)，计算每个 dp[i] 需 O(N)。
- 空间复杂度 O(N) ： dp 列表占用线性大小额外空间。
















