





<!-- 



[19. 删除链表的倒数第 N 个结点](https://leetcode-cn.com/problems/remove-nth-node-from-end-of-list/)

[8. 字符串转换整数 (atoi)](https://leetcode-cn.com/problems/string-to-integer-atoi/) 

[2. 两数相加](https://leetcode-cn.com/problems/add-two-numbers/)

[144. 二叉树的前序遍历](https://leetcode-cn.com/problems/binary-tree-preorder-traversal/)

[148. 排序链表](https://leetcode-cn.com/problems/sort-list/)

[72. 编辑距离](https://leetcode-cn.com/problems/edit-distance/) 

[105. 从前序与中序遍历序列构造二叉树](https://leetcode-cn.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/)

[151. 翻转字符串里的单词](https://leetcode-cn.com/problems/reverse-words-in-a-string/)

[76. 最小覆盖子串](https://leetcode-cn.com/problems/minimum-window-substring/) 

[31. 下一个排列](https://leetcode-cn.com/problems/next-permutation/)






[1143. 最长公共子序列](https://leetcode-cn.com/problems/longest-common-subsequence/)

[104. 二叉树的最大深度](https://leetcode-cn.com/problems/maximum-depth-of-binary-tree/) 




[110. 平衡二叉树](https://leetcode-cn.com/problems/balanced-binary-tree/)



[718. 最长重复子数组](https://leetcode-cn.com/problems/maximum-length-of-repeated-subarray/)





[113. 路径总和 II](https://leetcode-cn.com/problems/path-sum-ii/)

[155. 最小栈](https://leetcode-cn.com/problems/min-stack/)





[234. 回文链表](https://leetcode-cn.com/problems/palindrome-linked-list/)

[169. 多数元素](https://leetcode-cn.com/problems/majority-element/)

[876. 链表的中间结点](https://leetcode-cn.com/problems/middle-of-the-linked-list/) 补充

[470. 用 Rand7() 实现 Rand10()](https://leetcode-cn.com/problems/implement-rand10-using-rand7/)



[226. 翻转二叉树](https://leetcode-cn.com/problems/invert-binary-tree/)

[543. 二叉树的直径](https://leetcode-cn.com/problems/diameter-of-binary-tree/)

[101. 对称二叉树](https://leetcode-cn.com/problems/symmetric-tree/)

[4. 寻找两个正序数组的中位数](https://leetcode-cn.com/problems/median-of-two-sorted-arrays/) 	




 -->





------









```go
func minWindow(s string, t string) string {
	need, window := map[byte]int{}, map[byte]int{}
	for i := range t {
		need[t[i]]++
	} // 记录最小覆盖子串的起始索引及长度
	left, right, valid, index, length := 0, 0, 0, 0, math.MaxInt64
	for right < len(s) {
		b := s[right]             // b 是将移入窗口的字符
		right++                   // 右移窗口
		if _, ok := need[b]; ok { // 进行窗口内数据的一系列更新
			window[b]++
			if window[b] == need[b] {
				valid++
			}
		}
		for valid == len(need) { // 判断左侧窗口是否要收缩
			if right-left < length { // 在这里更新最小覆盖子串
				index = left
				length = right - left
			}
			d := s[left]              // d 是将移出窗口的字符
			left++                    // 左移窗口
			if _, ok := need[d]; ok { // 进行窗口内数据的一系列更新
				window[d]--
				if window[d] < need[d] {
					valid--
				}
			}
		}
	}
	if length == math.MaxInt64 {
		return ""
	}
	return s[index : index+length]
}
```

1. Use two pointers: start and end to represent a window.
2. Move end to find a valid window.
3. When a valid window is found, move start to find a smaller window.

```go
func minWindow(s string, t string) string {
	need := make(map[byte]int)
	for i := range t {
		need[t[i]]++
	}
	start, end, count, i := 0, -1, len(t), 0
	for j := 0; j < len(s); j++ {
		if need[s[j]] > 0 { //如果t中存在字符 s[j]，减少计数器
			count--
		}
		need[s[j]]--    //减少s[j]，如果字符s[j]在t中不存在，need[s[j]]置为负数
		if count == 0 { //找到有效的窗口后，开始移动以查找较小的窗口
			for i < j && need[s[i]] < 0 { //指针未越界且 字符s[i]在t中不存在
				need[s[i]]++ //移除t中不存在的字符 s[i]
				i++          // 左移窗口
			}
			if end == -1 || j-i < end-start {
				start, end = i, j
			}
			need[s[i]]++ //移除t中存在的字符 s[i]
			count++
			i++ //缩小窗口
		}
	}
	if end < start {
		return ""
	}
	return s[start : end+1]
}
```




## [2. 两数相加](https://leetcode-cn.com/problems/add-two-numbers/)

```go
func addTwoNumbers(l1 *ListNode, l2 *ListNode) *ListNode {
	dummy := new(ListNode)
	curr, carry := dummy, 0
	for l1 != nil || l2 != nil || carry > 0 {
		curr.Next = new(ListNode)
		curr = curr.Next
		if l1 != nil {
			carry += l1.Val
			l1 = l1.Next
		}
		if l2 != nil {
			carry += l2.Val
			l2 = l2.Next
		}
		curr.Val = carry % 10
		carry /= 10
	}
	return dummy.Next
}
```




## [151. 翻转字符串里的单词](https://leetcode-cn.com/problems/reverse-words-in-a-string/)

**解题思路** 

- 给出一个中间有空格分隔的字符串，要求把这个字符串按照单词的维度前后翻转。
- 依照题意，先把字符串按照空格分隔成每个小单词，然后把单词前后翻转，最后再把每个单词中间添加空格。

```go
func reverseWords(s string) string {
	ss := strings.Fields(s)
	reverse(&ss, 0, len(ss)-1)
	return strings.Join(ss, " ")
}
func reverse(m *[]string, i, j int) {
	for i <= j {
		(*m)[i], (*m)[j] = (*m)[j], (*m)[i]
		i++
		j--
	}
}
```

```go
func reverseWords(s string) string {
	ss := strings.Fields(s) //ss = ["the", "sky", "is", "blue"]
	var reverse func([]string, int, int)
	reverse = func(m []string, i, j int) {
		for i <= j {
			m[i], m[j] = m[j], m[i] // ["blue", "sky", "is", "the"]
			i++
			j--
		}
	}
	reverse(ss, 0, len(ss)-1)
	return strings.Join(ss, " ") //"blue is sky the"
}
```

**func Fields**

```go
func Fields(s string) []string
```
返回将字符串按照空白（unicode.IsSpace确定，可以是一到多个连续的空白字符）分割的多个字符串。如果字符串全部是空白或者是空字符串的话，会返回空切片。

```go
Example
fmt.Printf("Fields are: %q", strings.Fields("  foo bar  baz   "))
Output:

Fields are: ["foo" "bar" "baz"]
```

**func Join**

```go
func Join(a []string, sep string) string
```
将一系列字符串连接为一个字符串，之间用sep来分隔。

```go
Example
s := []string{"foo", "bar", "baz"}
fmt.Println(strings.Join(s, ", "))
Output:

foo, bar, baz
```


## [8. 字符串转换整数 (atoi)](https://leetcode-cn.com/problems/string-to-integer-atoi/) 

```go
func myAtoi(s string) int {
	abs, sign, i, n := 0, 1, 0, len(s)
	//丢弃无用的前导空格
	for i < n && s[i] == ' ' {
		i++
	}
	//标记正负号
	if i < n {
		if s[i] == '-' {
			sign = -1
			i++
		} else if s[i] == '+' {
			sign = 1
			i++
		}
	}
	for i < n && s[i] >= '0' && s[i] <= '9' {
		abs = 10*abs + int(s[i]-'0')  //字节 byte '0' == 48
		if sign*abs < math.MinInt32 { //整数超过 32 位有符号整数范围
			return math.MinInt32
		} else if sign*abs > math.MaxInt32 {
			return math.MaxInt32
		}
		i++
	}
	return sign * abs
}
```
复杂度分析

- 时间复杂度：O(n)，其中 n 为字符串的长度。我们只需要依次处理所有的字符，处理每个字符需要的时间为 O(1)。

- 空间复杂度：O(1)，只需要常数空间存储。








## [72. 编辑距离](https://leetcode-cn.com/problems/edit-distance/) 



```go
func minDistance(word1 string, word2 string) int {
	m, n := len(word1), len(word2)
	dp := make([][]int, m+1)
	for i := range dp {
		dp[i] = make([]int, n+1)
	}
	for i := 0; i < m+1; i++ {
		dp[i][0] = i // word1[i] 变成 word2[0], 删掉 word1[i], 需要 i 部操作
	}
	for j := 0; j < n+1; j++ {
		dp[0][j] = j // word1[0] 变成 word2[j], 插入 word1[j]，需要 j 部操作
	}
	for i := 1; i < m+1; i++ {
		for j := 1; j < n+1; j++ {
			if word1[i-1] == word2[j-1] {
				dp[i][j] = dp[i-1][j-1]
			} else { // Min(插入，删除，替换)
				dp[i][j] = Min(dp[i][j-1], dp[i-1][j], dp[i-1][j-1]) + 1
			}
		}
	}
	return dp[m][n]
}
func Min(args ...int) int {
	min := args[0]
	for _, item := range args {
		if item < min {
			min = item
		}
	}
	return min
}

```

![](https://pic.leetcode-cn.com/8704230781a0bc6f11ff317757c73505e8c4cb2c1ca1dcdfb9b0c84eb08d901f-%E5%B9%BB%E7%81%AF%E7%89%872.PNG)

![](https://pic.leetcode-cn.com/bfc8d2232a17c8999b7d700806bf0048ad4727b567ee756e01fa16750e9e0d07-%E5%B9%BB%E7%81%AF%E7%89%873.PNG)

![](https://pic.leetcode-cn.com/0e81d8994ffa586183a32f545c259f81d7b33baa753275a4ffb9587c65a55c15-%E5%B9%BB%E7%81%AF%E7%89%874.PNG)



**替换 word1[i] 字符：dp[i-1][j-1] + 1** 
	case 1. if word1[i] == word2[j]  相等跳过: dp[i-1][j-1] 不操作
	case 2. if word1[i] != word2[j]  word1[i-1] 等于 word2[j-1], 再加上最后一步替换操作(+1)  dp[i-1][j-1] + 1

**插入 word1[i] 字符：dp[i][j-1] + 1**
	word1[i] 插入一个字符变成 word2[j], 插完相等，因此插入字符一定是 word2[j]， 插入前 word1[i] 和 word2[j-1]已经匹配(相等)

**删除 word1[i] 字符：dp[i-1][j] + 1**
	word1[i] 删除一个字符变成 word2[j], word1[i] 删除前 word1[i-1] 和 word2[j] 已经匹配(相等)







## [104. 二叉树的最大深度](https://leetcode-cn.com/problems/maximum-depth-of-binary-tree/)

**方法一：深度优先搜索**

思路与算法

如果我们知道了左子树和右子树的最大深度 ll 和 rr，那么该二叉树的最大深度即为

max(l,r)+1

而左子树和右子树的最大深度又可以以同样的方式进行计算。因此我们可以用「深度优先搜索」的方法来计算二叉树的最大深度。具体而言，在计算当前二叉树的最大深度时，可以先递归计算出其左子树和右子树的最大深度，然后在 O(1) 时间内计算出当前二叉树的最大深度。递归在访问到空节点时退出。
![](https://assets.leetcode-cn.com/solution-static/104/7.png)
![](https://assets.leetcode-cn.com/solution-static/104/10.png)

```go
func maxDepth(root *TreeNode) int {
	if root == nil {
		return 0
	}
	return max(maxDepth(root.Left), maxDepth(root.Right)) + 1
}
func max(x, y int) int {
	if x > y {
		return x
	}
	return y
}
```

复杂度分析

- 时间复杂度：O(n)，其中 n 为二叉树节点的个数。每个节点在递归中只被遍历一次。
- 空间复杂度：O(height)，其中 height 表示二叉树的高度。递归函数需要栈空间，而栈空间取决于递归的深度，因此空间复杂度等价于二叉树的高度。





## [110. 平衡二叉树](https://leetcode-cn.com/problems/balanced-binary-tree/)

**方法一：自顶向下的递归 (前序遍历)**

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


**方法二：自底向上的递归 (后序遍历)**

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







## [144. 二叉树的前序遍历](https://leetcode-cn.com/problems/binary-tree-preorder-traversal/)

**方法一：递归**

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

**方法二：迭代**


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









## [105. 从前序与中序遍历序列构造二叉树](https://leetcode-cn.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/)


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



## [113. 路径总和 II](https://leetcode-cn.com/problems/path-sum-ii/)

**方法一：深度优先搜索**

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







## [155. 最小栈](https://leetcode-cn.com/problems/min-stack/)

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






## [1143. 最长公共子序列](https://leetcode-cn.com/problems/longest-common-subsequence/)

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




## [19. 删除链表的倒数第 N 个结点](https://leetcode-cn.com/problems/remove-nth-node-from-end-of-list/)

**方法一：双指针**
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





## [470. 用 Rand7() 实现 Rand10()](https://leetcode-cn.com/problems/implement-rand10-using-rand7/)

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


## [543. 二叉树的直径](https://leetcode-cn.com/problems/diameter-of-binary-tree/)

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












## [101. 对称二叉树](https://leetcode-cn.com/problems/symmetric-tree/)

**方法一：递归**
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



## [4. 寻找两个正序数组的中位数](https://leetcode-cn.com/problems/median-of-two-sorted-arrays/)


for example，a=[1 2 3 4 6 9]and, b=[1 1 5 6 9 10 11]，total numbers are 13， 
you should find the seventh number , int(7/2)=3, a[3]<b[3], 
so you don't need to consider a[0],a[1],a[2] because they can't be the seventh number. Then find the fourth number in the others numbers which don't include a[0]a[1]a[2]. just like this , decrease half of numbers every time .......


```go
func findMedianSortedArrays(nums1 []int, nums2 []int) float64 {
	if l := len(nums1) + len(nums2); l%2 == 0 {
		return (findKth(nums1, nums2, l/2-1) + findKth(nums1, nums2, l/2)) / 2.0
	} else {
		return findKth(nums1, nums2, l/2)
	}
}
func findKth(nums1, nums2 []int, k int) float64 {
	for {
		l1, l2 := len(nums1), len(nums2)
		m1, m2 := l1/2, l2/2
		if l1 == 0 {
			return float64(nums2[k])
		} else if l2 == 0 {
			return float64(nums1[k])
		} else if k == 0 {
			if n1, n2 := nums1[0], nums2[0]; n1 <= n2 {
				return float64(n1)
			} else {
				return float64(n2)
			}
		}
		if k <= m1+m2 {
			if nums1[m1] <= nums2[m2] {
				nums2 = nums2[:m2]
			} else {
				nums1 = nums1[:m1]
			}
		} else {
			if nums1[m1] <= nums2[m2] {
				nums1 = nums1[m1+1:]
				k -= m1 + 1
			} else {
				nums2 = nums2[m2+1:]
			}
		}
	}
}

```
复杂度分析

- 时间复杂度：O(log(m+n))，其中 m 和 n 分别是数组 nums1 和 nums2 的长度。初始时有 k=(m+n)/2 或 k=(m+n)/2+1，每一轮循环可以将查找范围减少一半，因此时间复杂度是 O(log(m+n))。

- 空间复杂度：O(1)。



```go
func findMedianSortedArrays(nums1 []int, nums2 []int) float64 {
	l1, l2 := len(nums1), len(nums2)
	if l1 > l2 {
		return findMedianSortedArrays(nums2, nums1)
	}
	for start, end := 0, l1; ; {
		nums1Med := (start + end) / 2
		nums2Med := (l2+l1+1)/2 - nums1Med
		nums1Left, nums1Right, nums2Left, nums2Right := math.MinInt64,
			math.MaxInt64, math.MinInt64, math.MaxInt64
		if nums1Med != 0 {
			nums1Left = nums1[nums1Med-1]
		}
		if nums1Med != l1 {
			nums1Right = nums1[nums1Med]
		}
		if nums2Med != 0 {
			nums2Left = nums2[nums2Med-1]
		}
		if nums2Med != l2 {
			nums2Right = nums2[nums2Med]
		}
		if nums1Left > nums2Right {
			end = nums1Med - 1
		} else if nums2Left > nums1Right {
			start = nums1Med + 1
		} else {
			if (l1+l2)%2 == 1 {
				return math.Max(float64(nums1Left), float64(nums2Left))
			}
			return (math.Max(float64(nums1Left), float64(nums2Left)) +
				math.Min(float64(nums1Right), float64(nums2Right))) / 2
		}
	}
}
```

复杂度分析

- 时间复杂度：O(logmin(m,n)))，其中 m 和 n 分别是数组 nums1 和 nums2 的长度。查找的区间是 [0,m]，而该区间的长度在每次循环之后都会减少为原来的一半。所以，只需要执行 logm 次循环。由于每次循环中的操作次数是常数，所以时间复杂度为 O(logm)。由于我们可能需要交换 nums1 和 nums2 使得 m≤n，因此时间复杂度是 O(log -min(m,n)))。

- 空间复杂度：O(1)。







## [148. 排序链表](https://leetcode-cn.com/problems/sort-list/)


**方法一：自底向上归并排序**

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


**解释**

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

**C++ 代码**
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

**方法二：自顶向下归并排序**

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
**复杂度分析**

- 时间复杂度：O(nlogn)，其中 n 是链表的长度。

- 空间复杂度：O(logn)，其中 n 是链表的长度。空间复杂度主要取决于递归调用的栈空间。



![1.png](http://ww1.sinaimg.cn/large/007daNw2ly1gph68oyr9nj30z70qb0tr.jpg)




## [31. 下一个排列](https://leetcode-cn.com/problems/next-permutation/)

**思路**

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


