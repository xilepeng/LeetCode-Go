
[543. 二叉树的直径](https://leetcode-cn.com/problems/diameter-of-binary-tree/)

[4. 寻找两个正序数组的中位数](https://leetcode-cn.com/problems/median-of-two-sorted-arrays/) 	next

[148. 排序链表](https://leetcode-cn.com/problems/sort-list/)

[470. 用 Rand7() 实现 Rand10()](https://leetcode-cn.com/problems/implement-rand10-using-rand7/)

[143. 重排链表](https://leetcode-cn.com/problems/reorder-list/)

[113. 路径总和 II](https://leetcode-cn.com/problems/path-sum-ii/)

[1143. 最长公共子序列](https://leetcode-cn.com/problems/longest-common-subsequence/)

[240. 搜索二维矩阵 II](https://leetcode-cn.com/problems/search-a-2d-matrix-ii/)

[98. 验证二叉搜索树](https://leetcode-cn.com/problems/validate-binary-search-tree/)

[39. 组合总和](https://leetcode-cn.com/problems/combination-sum/)

[239. 滑动窗口最大值](https://leetcode-cn.com/problems/sliding-window-maximum/)


------


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

[4. 寻找两个正序数组的中位数](https://leetcode-cn.com/problems/median-of-two-sorted-arrays/)	next






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

