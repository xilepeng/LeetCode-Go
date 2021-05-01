

[160. 相交链表](https://leetcode-cn.com/problems/intersection-of-two-linked-lists/)

[415. 字符串相加](https://leetcode-cn.com/problems/add-strings/)

[141. 环形链表](https://leetcode-cn.com/problems/linked-list-cycle/)

[121. 买卖股票的最佳时机](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock/)

[102. 二叉树的层序遍历](https://leetcode-cn.com/problems/binary-tree-level-order-traversal/)

[103. 二叉树的锯齿形层序遍历](https://leetcode-cn.com/problems/binary-tree-zigzag-level-order-traversal/)

[88. 合并两个有序数组](https://leetcode-cn.com/problems/merge-sorted-array/)

[236. 二叉树的最近公共祖先](https://leetcode-cn.com/problems/lowest-common-ancestor-of-a-binary-tree/)

[20. 有效的括号](https://leetcode-cn.com/problems/valid-parentheses/)

[142. 环形链表 II](https://leetcode-cn.com/problems/linked-list-cycle-ii/)

------



[160. 相交链表](https://leetcode-cn.com/problems/intersection-of-two-linked-lists/)

### 方法一：双指针法

![截屏2021-04-21 11.23.18.png](http://ww1.sinaimg.cn/large/007daNw2ly1gpr7bhprljj31ck0p6do3.jpg)

![截屏2021-04-21 11.23.31.png](http://ww1.sinaimg.cn/large/007daNw2ly1gpr7brsn7wj31c60m0793.jpg)

```go
func getIntersectionNode(headA, headB *ListNode) *ListNode {
	A, B := headA, headB
	for A != B {
		if A != nil {
			A = A.Next
		} else {
			A = headB
		}
		if B != nil {
			B = B.Next
		} else {
			B = headA
		}
	}
	return A
}
```

复杂度分析

- 时间复杂度 : O(m+n)。
- 空间复杂度 : O(1)。


[415. 字符串相加](https://leetcode-cn.com/problems/add-strings/)

```go
func addStrings(num1 string, num2 string) string {
	carry := 0
	res := ""
	for i, j := len(num1)-1, len(num2)-1; i >= 0 || j >= 0 || carry != 0; i, j = i-1, j-1 {
		var x, y int
		if i >= 0 {
			x = int(num1[i] - '0')
		}
		if j >= 0 {
			y = int(num2[j] - '0')
		}
		tmp := x + y + carry
		res = strconv.Itoa(tmp%10) + res
		carry = tmp / 10
	}
	return res
}
```

[141. 环形链表](https://leetcode-cn.com/problems/linked-list-cycle/)

![截屏2021-04-21 12.13.52.png](http://ww1.sinaimg.cn/large/007daNw2ly1gpr8r3r6ijj31780mcq9j.jpg)

```go
func hasCycle(head *ListNode) bool {
	if head == nil || head.Next == nil {
		return false
	}
	slow, fast := head, head.Next
	for slow != fast {
		if fast == nil || fast.Next == nil {
			return false
		}
		slow = slow.Next
		fast = fast.Next.Next
	}
	return true
}
```




[121. 买卖股票的最佳时机](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock/)

```go
func maxProfit(prices []int) int {
	minprice := math.MaxInt32
	maxprofit := 0
	for _, price := range prices {
		if price < minprice {
			minprice = price
		} else if maxprofit < price-minprice {
			maxprofit = price - minprice
		}
	}
	return maxprofit
}
```

```go
func maxProfit(prices []int) int {
	minprice := math.MaxInt32
	maxprofit := 0
	for _, price := range prices {
		maxprofit = Max(price-minprice, maxprofit)
		minprice = Min(price, minprice)
	}
	return maxprofit
}
func Max(x, y int) int {
	if x > y {
		return x
	}
	return y
}
func Min(x, y int) int {
	if x < y {
		return x
	}
	return y
}
```

```go
func maxProfit(prices []int) int {
	if prices == nil || len(prices) == 0 {
		return 0
	}
	minprice := prices[0]
	maxprofit := 0
	for i := 1; i < len(prices); i++ {
		if prices[i] < minprice {
			minprice = prices[i]
		}
		if maxprofit < prices[i]-minprice {
			maxprofit = prices[i] - minprice
		}
	}
	return maxprofit
}
```

```go
func maxProfit(prices []int) int {
	maxprofit := 0
	for i := 0; i < len(prices)-1; i++ {
		for j := i + 1; j < len(prices); j++ {
			profit := prices[j] - prices[i]
			if maxprofit < profit {
				maxprofit = profit
			}
		}
	}
	return maxprofit
}
```
Time Limit Exceeded





[102. 二叉树的层序遍历](https://leetcode-cn.com/problems/binary-tree-level-order-traversal/)

方法一：DFS递归

```go
var res [][]int

func levelOrder(root *TreeNode) [][]int {
	res = [][]int{}
	dfs(root, 0)
	return res
}
func dfs(root *TreeNode, level int) {
	if root != nil {
		if level == len(res) {
			res = append(res, []int{})
		}
		res[level] = append(res[level], root.Val)
		dfs(root.Left, level+1)
		dfs(root.Right, level+1)
	}
```

方法二：BFS(queue)迭代

```go
func levelOrder(root *TreeNode) [][]int {
	res := [][]int{}
	if root == nil {
		return res
	}
	queue := []*TreeNode{root}
	for level := 0; 0 < len(queue); level++ {
		res = append(res, []int{})
		next := []*TreeNode{}
		for j := 0; j < len(queue); j++ {
			node := queue[j]
			res[level] = append(res[level], node.Val)
			if node.Left != nil {
				next = append(next, node.Left)
			}
			if node.Right != nil {
				next = append(next, node.Right)
			}
		}
		queue = next
	}
	return res
}
```



[103. 二叉树的锯齿形层序遍历](https://leetcode-cn.com/problems/binary-tree-zigzag-level-order-traversal/)

方法一：深度优先遍历
```go
var res [][]int

func zigzagLevelOrder(root *TreeNode) [][]int {
	res = [][]int{}
	dfs(root, 0)
	return res
}
func dfs(root *TreeNode, level int) {
	if root != nil {
		if len(res) == level {
			res = append(res, []int{})
		}
		if level%2 == 0 {//偶数层，从左往右
			res[level] = append(res[level], root.Val)
		} else {//奇数层，从右往左,翻转
			res[level] = append([]int{root.Val}, res[level]...)
		}
		dfs(root.Left, level+1)
		dfs(root.Right, level+1)
	}
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
func zigzagLevelOrder(root *TreeNode) [][]int {
	res := [][]int{}
	var dfs func(*TreeNode, int)

	dfs = func(root *TreeNode, level int) {
		if root != nil {
			if len(res) == level {
				res = append(res, []int{})
			}
			if level%2 == 0 {
				res[level] = append(res[level], root.Val)
			} else {//翻转奇数层的元素
				res[level] = append([]int{root.Val}, res[level]...)
			}
			dfs(root.Left, level+1)
			dfs(root.Right, level+1)
		}
	}

	dfs(root, 0)
	return res
}
```

方法二：广度优先遍历

```go
/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */
func zigzagLevelOrder(root *TreeNode) [][]int {
	res := [][]int{}
	if root == nil {
		return res
	}
	queue := []*TreeNode{root}
	for level := 0; len(queue) > 0; level++ {
		res = append(res, []int{})
		next, q := []int{}, queue
		queue = nil
		for _, node := range q {
			next = append(next, node.Val)
			if node.Left != nil {
				queue = append(queue, node.Left)
			}
			if node.Right != nil {
				queue = append(queue, node.Right)
			}
		}
		if level%2 == 1 { //翻转奇数层的元素
			for i, n := 0, len(next); i < n/2; i++ {
				next[i], next[n-1-i] = next[n-1-i], next[i]
			}
		}
		res[level] = append(res[level], next...)
	}
	return res
}
```



[88. 合并两个有序数组](https://leetcode-cn.com/problems/merge-sorted-array/)

```go
func merge(nums1 []int, m int, nums2 []int, n int) {
	i, j := m-1, n-1
	for k := m + n - 1; k >= 0; k-- {
		if j < 0 || (i >= 0 && nums1[i] > nums2[j]) {
			nums1[k] = nums1[i]
			i--
		} else {
			nums1[k] = nums2[j]
			j--
		}
	}
}
```


[236. 二叉树的最近公共祖先](https://leetcode-cn.com/problems/lowest-common-ancestor-of-a-binary-tree/)

```go
func lowestCommonAncestor(root, p, q *TreeNode) *TreeNode {
	if root == nil {
		return nil
	}
	if root.Val == p.Val || root.Val == q.Val {
		return root
	}
	left := lowestCommonAncestor(root.Left, p, q)
	right := lowestCommonAncestor(root.Right, p, q)
	if left != nil && right != nil {
		return root
	}
	if left == nil {
		return right
	}
	return left
}
```


[20. 有效的括号](https://leetcode-cn.com/problems/valid-parentheses/)

```go
func isValid(s string) bool {
	if len(s) == 0 {
		return true
	}
	stack := make([]rune, 0)
	for _, v := range s {
		if v == '(' || v == '{' || v == '[' {
			stack = append(stack, v)
		} else if v == ')' && len(stack) > 0 && stack[len(stack)-1] == '(' ||
			v == '}' && len(stack) > 0 && stack[len(stack)-1] == '{' ||
			v == ']' && len(stack) > 0 && stack[len(stack)-1] == '[' {
			stack = stack[:len(stack)-1]
		} else {
			return false
		}
	}
	return len(stack) == 0
}
```

[142. 环形链表 II](https://leetcode-cn.com/problems/linked-list-cycle-ii/)

方法一：快慢指针
我们使用两个指针，fast 与 slow。它们起始都位于链表的头部。随后，slow 指针每次向后移动一个位置，而 fast 指针向后移动两个位置。如果链表中存在环，则 fast 指针最终将再次与 slow 指针在环中相遇。


```go
func detectCycle(head *ListNode) *ListNode {
	slow, fast := head, head
	for fast != nil {
		slow = slow.Next
		if fast.Next == nil {
			return nil
		}
		fast = fast.Next.Next
		if slow == fast { //第一次相遇
			p := head
			for p != slow {
				p = p.Nextx
				slow = slow.Next
			}
			return p
		}
	}
	return nil
}
```
