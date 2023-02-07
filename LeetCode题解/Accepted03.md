
1. [✅ 19. 删除链表的倒数第 N 个结点](#-19-删除链表的倒数第-n-个结点)
2. [✅ 8. 字符串转换整数 (atoi)](#-8-字符串转换整数-atoi)
3. [✅ 2. 两数相加](#-2-两数相加)
4. [✅ 144. 二叉树的前序遍历](#-144-二叉树的前序遍历)
5. [✅ 148. 排序链表](#-148-排序链表)
6. [72. 编辑距离](#72-编辑距离)
7. [105. 从前序与中序遍历序列构造二叉树](#105-从前序与中序遍历序列构造二叉树)
8. [151. 翻转字符串里的单词](#151-翻转字符串里的单词)
9. [76. 最小覆盖子串](#76-最小覆盖子串)
10. [31. 下一个排列](#31-下一个排列)
11. [✅ 1143. 最长公共子序列](#-1143-最长公共子序列)
12. [✅ 4. 寻找两个正序数组的中位数](#-4-寻找两个正序数组的中位数)
13. [✅ 104. 二叉树的最大深度](#-104-二叉树的最大深度)
14. [✅ 129. 求根节点到叶节点数字之和](#-129-求根节点到叶节点数字之和)
15. [✅ 110. 平衡二叉树](#-110-平衡二叉树)
16. [✅ 239. 滑动窗口最大值](#-239-滑动窗口最大值)
17. [✅ 93. 复原 IP 地址](#-93-复原-ip-地址)
18. [✅ 113. 路径总和 II](#-113-路径总和-ii)
19. [41. 缺失的第一个正数](#41-缺失的第一个正数)
20. [155. 最小栈](#155-最小栈)



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
-->










## ✅ [19. 删除链表的倒数第 N 个结点](https://leetcode-cn.com/problems/remove-nth-node-from-end-of-list/)

**方法一：双指针**

思路与算法

使用两个指针 first 和 second 同时对链表进行遍历，并且 first 比 second 超前 n 个节点。当 first 遍历到链表的末尾时，second 就恰好处于倒数第 n 个节点。


![](https://assets.leetcode-cn.com/solution-static/19/p3.png)

``` go
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


``` go
/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */
func removeNthFromEnd(head *ListNode, n int) *ListNode {
	dummy := &ListNode{0, head}
	slow, fast := dummy, head
	for i := 0; fast != nil; i++ {
		if i >= n {
			slow = slow.Next
		}
		fast = fast.Next
	}
	slow.Next = slow.Next.Next
	return dummy.Next
}
```

``` go
/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */
func removeNthFromEnd(head *ListNode, n int) *ListNode {
	dummy := &ListNode{0, head}
	slow, fast := dummy, head
	for i := 0; fast != nil; fast = fast.Next {
		if i >= n {
			slow = slow.Next
		}
		i++
	}
	slow.Next = slow.Next.Next
	return dummy.Next
}
```




## ✅ [8. 字符串转换整数 (atoi)](https://leetcode-cn.com/problems/string-to-integer-atoi/) 

``` go
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












## ✅ [2. 两数相加](https://leetcode-cn.com/problems/add-two-numbers/)

``` go
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







## ✅ [144. 二叉树的前序遍历](https://leetcode-cn.com/problems/binary-tree-preorder-traversal/)

**方法一：递归**

``` go
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

``` go
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

``` go
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

``` go
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


``` go
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

``` go
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




## ✅ [148. 排序链表](https://leetcode-cn.com/problems/sort-list/)


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



``` go
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

``` go
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
``` go
func sortList (head) {
	对链表进行二分
	l = sortList(左链) // 已排序的左链
	r = sortList(右链) // 已排序的右链
	merged = mergeList(l, r) // 进行合并
	return merged		// 返回合并的结果给父调用
}
```

![3.png](http://ww1.sinaimg.cn/large/007daNw2ly1gph6854l3wj31i80luwj5.jpg)

**二分、merge 函数**
- 二分，用快慢指针法，快指针走两步，慢指针走一步，快指针越界时，慢指针正好到达中点，只需记录慢指针的前一个指针，就能断成两链。
- merge 函数做的事是合并两个有序的左右链
    1. 设置虚拟头结点，用 prev 指针去“穿针引线”，prev 初始指向 dummy
    2. 每次都确定 prev.Next 的指向，并注意 l1，l2指针的推进，和 prev 指针的推进
    3. 最后返回 dummy.Next ，即合并后的链。


![4.png](http://ww1.sinaimg.cn/large/007daNw2ly1gph68ecpsoj31gc0iq43c.jpg)


``` go
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

``` go
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









## [72. 编辑距离](https://leetcode-cn.com/problems/edit-distance/) 



``` go
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








## [105. 从前序与中序遍历序列构造二叉树](https://leetcode-cn.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/)


``` go
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
	root := &TreeNode{Val: preorder[0]}
	for pos, node_val := range inorder {
		if node_val == root.Val {
			root.Left = buildTree(preorder[1:pos+1], inorder[:pos+1])
			root.Right = buildTree(preorder[pos+1:], inorder[pos+1:])
		}
	}
	return root
}
```









## [151. 翻转字符串里的单词](https://leetcode-cn.com/problems/reverse-words-in-a-string/)

**解题思路** 

- 给出一个中间有空格分隔的字符串，要求把这个字符串按照单词的维度前后翻转。
- 依照题意，先把字符串按照空格分隔成每个小单词，然后把单词前后翻转，最后再把每个单词中间添加空格。

``` go
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

``` go
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

``` go
func Fields(s string) []string
```
返回将字符串按照空白（unicode.IsSpace确定，可以是一到多个连续的空白字符）分割的多个字符串。如果字符串全部是空白或者是空字符串的话，会返回空切片。

``` go
Example
fmt.Printf("Fields are: %q", strings.Fields("  foo bar  baz   "))
Output:

Fields are: ["foo" "bar" "baz"]
```

**func Join**

``` go
func Join(a []string, sep string) string
```
将一系列字符串连接为一个字符串，之间用sep来分隔。

``` go
Example
s := []string{"foo", "bar", "baz"}
fmt.Println(strings.Join(s, ", "))
Output:

foo, bar, baz
```







## [76. 最小覆盖子串](https://leetcode-cn.com/problems/minimum-window-substring/) 

``` go
func minWindow(s string, t string) string {
	need, window := map[byte]int{}, map[byte]int{}
	for i := range t {
		need[t[i]]++  // 记录字符频数
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

``` go
func minWindow(s string, t string) string {
	need := make(map[byte]int)
	for i := range t {
		need[t[i]]++ // 记录字符频数
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








## [31. 下一个排列](https://leetcode-cn.com/problems/next-permutation/)

**思路**

1. 我们需要将一个左边的「较小数」与一个右边的「较大数」交换，以能够让当前排列变大，从而得到下一个排列。

2. 同时我们要让这个「较小数」尽量靠右，而「较大数」尽可能小。当交换完成后，「较大数」右边的数需要按照升序重新排列。这样可以在保证新排列大于原来排列的情况下，使变大的幅度尽可能小。


- 从低位挑一个大一点的数，换掉前面的小一点的一个数，实现变大。
- 变大的幅度要尽量小。


像 [3,2,1] 递减的，没有下一个排列，因为大的已经尽量往前排了，没法更大。





``` go
func nextPermutation(nums []int) {
	i := len(nums) - 2                   // 从右向左遍历，i从倒数第二开始是为了nums[i+1]要存在
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




---




<!-- 

[1143. 最长公共子序列](https://leetcode-cn.com/problems/longest-common-subsequence/)

[4. 寻找两个正序数组的中位数](https://leetcode-cn.com/problems/median-of-two-sorted-arrays/) 	

[104. 二叉树的最大深度](https://leetcode-cn.com/problems/maximum-depth-of-binary-tree/) 

[129. 求根节点到叶节点数字之和](https://leetcode-cn.com/problems/sum-root-to-leaf-numbers/)

[110. 平衡二叉树](https://leetcode-cn.com/problems/balanced-binary-tree/)

[239. 滑动窗口最大值](https://leetcode-cn.com/problems/sliding-window-maximum/)

[93. 复原 IP 地址](https://leetcode-cn.com/problems/restore-ip-addresses/) 

[113. 路径总和 II](https://leetcode-cn.com/problems/path-sum-ii/)

[41. 缺失的第一个正数](https://leetcode-cn.com/problems/first-missing-positive/)

[155. 最小栈](https://leetcode-cn.com/problems/min-stack/)

 -->







## ✅ [1143. 最长公共子序列](https://leetcode-cn.com/problems/longest-common-subsequence/)


``` go
func longestCommonSubsequence(text1 string, text2 string) int {
	m, n := len(text1), len(text2)
	dp := make([][]int, m+1) // 创建二维数组
	for i := range dp {
		dp[i] = make([]int, n+1)
	}
	for i, c1 := range text1 {
		for j, c2 := range text2 {
			if c1 == c2 {
				dp[i+1][j+1] = dp[i][j] + 1
			} else {
				dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])
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


状态转移方程：

![截屏2021-04-15 21.12.08.png](http://ww1.sinaimg.cn/large/007daNw2ly1gpkqtiu1u5j30t404it9a.jpg)

最终计算得到 dp[m][n] 即为 text1 和 text2 的最长公共子序列的长度。

![](https://pic.leetcode-cn.com/1617411822-KhEKGw-image.png)

``` go
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

[参考](https://leetcode-cn.com/problems/longest-common-subsequence/solution/zui-chang-gong-gong-zi-xu-lie-by-leetcod-y7u0/)





## ✅ [4. 寻找两个正序数组的中位数](https://leetcode-cn.com/problems/median-of-two-sorted-arrays/)


for example，a=[1 2 3 4 6 9]and, b=[1 1 5 6 9 10 11]，total numbers are 13， 
you should find the seventh number , int(7/2)=3, a[3]<b[3], 
so you don't need to consider a[0],a[1],a[2] because they can't be the seventh number. Then find the fourth number in the others numbers which don't include a[0]a[1]a[2]. just like this , decrease half of numbers every time .......


``` go
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



``` go
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






## ✅ [104. 二叉树的最大深度](https://leetcode-cn.com/problems/maximum-depth-of-binary-tree/)

**方法一：深度优先搜索**

思路与算法

如果我们知道了左子树和右子树的最大深度 ll 和 rr，那么该二叉树的最大深度即为

max(l,r)+1

而左子树和右子树的最大深度又可以以同样的方式进行计算。因此我们可以用「深度优先搜索」的方法来计算二叉树的最大深度。具体而言，在计算当前二叉树的最大深度时，可以先递归计算出其左子树和右子树的最大深度，然后在 O(1) 时间内计算出当前二叉树的最大深度。递归在访问到空节点时退出。
![](https://assets.leetcode-cn.com/solution-static/104/7.png)
![](https://assets.leetcode-cn.com/solution-static/104/10.png)


``` go
/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */
func maxDepth(root *TreeNode) int {
	if root == nil {
		return 0
	}
	left_depth := maxDepth(root.Left)
	right_depth := maxDepth(root.Right)
	return max(left_depth, right_depth) + 1
}
func max(x, y int) int {
	if x > y {
		return x
	}
	return y
}
```

``` go
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








## ✅ [129. 求根节点到叶节点数字之和](https://leetcode-cn.com/problems/sum-root-to-leaf-numbers/)

**方法一：深度优先搜索**
思路与算法

从根节点开始，遍历每个节点，如果遇到叶子节点，则将叶子节点对应的数字加到数字之和。如果当前节点不是叶子节点，则计算其子节点对应的数字，然后对子节点递归遍历。

![](https://assets.leetcode-cn.com/solution-static/129/fig1.png)

![](https://pic.leetcode-cn.com/1603933660-UNWQbT-image.png)

``` go
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






## ✅ [110. 平衡二叉树](https://leetcode-cn.com/problems/balanced-binary-tree/)

**方法一：自顶向下的递归 (前序遍历)**

具体做法类似于二叉树的前序遍历，即对于当前遍历到的节点，首先计算左右子树的高度，如果左右子树的高度差是否不超过 11，再分别递归地遍历左右子节点，并判断左子树和右子树是否平衡。这是一个自顶向下的递归的过程。


``` go
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

``` go
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





## ✅ [239. 滑动窗口最大值](https://leetcode-cn.com/problems/sliding-window-maximum/)

**方法一 暴力解法 O(nk)**

``` go
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

**方法二 双端队列 Deque**

最优的解法是用双端队列，队列的一边永远都存的是窗口的最大值，队列的另外一个边存的是比最大值小的值。队列中最大值左边的所有值都出队。在保证了双端队列的一边即是最大值以后，

- 时间复杂度是 O(n)
- 空间复杂度是 O(K)

``` go
// 维护单调递减队列
func maxSlidingWindow(nums []int, k int) []int {
	q, res := []int{}, []int{}
	for i, v := range nums {
		if i >= k && q[0] <= i-k { // 队满
			q = q[1:] // 删除队头
		}
		for len(q) > 0 && nums[q[len(q)-1]] <= v { // 队尾元素小于等于当前元素
			q = q[:len(q)-1] // 删除队尾
		}
		q = append(q, i) // 存储当前索引
		if i >= k-1 {
			res = append(res, nums[q[0]])
		}
	}
	return res
}
```


![](http://ww1.sinaimg.cn/large/007daNw2ly1gpmcfyuvh1j319g0mun0a.jpg)

``` go
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







## ✅ [93. 复原 IP 地址](https://leetcode-cn.com/problems/restore-ip-addresses/)

思路

- 以 "25525511135" 为例，做第一步时我们有几种选择？

1. 选 "2" 作为第一个片段
2. 选 "25" 作为第一个片段
3. 选 "255" 作为第一个片段
- 能切三种不同的长度，切第二个片段时，又面临三种选择。
- 这会向下分支形成一棵树，我们用 DFS 去遍历所有选择，必要时提前回溯。
	因为某一步的选择可能是错的，得不到正确的结果，不要往下做了。撤销最后一个选择，回到选择前的状态，去试另一个选择。
- 回溯的第一个要点：选择，它展开了一颗空间树。

****回溯的要点二——约束****

- 约束条件限制了当前的选项，这道题的约束条件是：
1. 一个片段的长度是 1~3
2. 片段的值范围是 0~255
3. 不能是 "0x"、"0xx" 形式（测试用例告诉我们的）
- 用这些约束进行充分地剪枝，去掉一些选择，避免搜索「不会产生正确答案」的分支。
  
**回溯的要点三——目标**

- 目标决定了什么时候捕获答案，什么时候砍掉死支，回溯。
- 目标是生成 4 个有效片段，并且要耗尽 IP 的字符。
- 当满足该条件时，说明生成了一个有效组合，加入解集，结束当前递归，继续探索别的分支。
- 如果满4个有效片段，但没耗尽字符，不是想要的解，不继续往下递归，提前回溯。
  
**定义 dfs 函数**
- dfs 函数传什么？也就是，用什么描述一个节点的状态？
- 选择切出一个片段后，继续递归剩余子串。可以传子串，也可以传指针，加上当前的片段数组，描述节点的状态。
- dfs 函数做的事：复原从 start 到末尾的子串。

我把递归树画了出来，可以看看回溯的细节：


![](https://pic.leetcode-cn.com/5276b1631cb1fc47d8d88dd021f1302213291bf05bfdfdc6209370ce9034be83-image.png)

如图['2','5','5','2']未耗尽字符，不是有效组合，不继续选下去。撤销选择"2"，回到之前的状态（当前分支砍掉了），切入到另一个分支，选择"25"。

回溯会穷举所有节点，通常用于解决「找出所有可能的组合」问题。

下图展示找到一个有效的组合的样子。start 指针越界，代表耗尽了所有字符，且满 4 个片段。


![](https://pic.leetcode-cn.com/e3e3a6dac1ecb79da18740f7968a5eedaa80d5a0e0e45463c7096f663748e0fa-image.png)

``` go
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


[参考](https://leetcode-cn.com/problems/restore-ip-addresses/solution/shou-hua-tu-jie-huan-yuan-dfs-hui-su-de-xi-jie-by-/)



























## ✅ [113. 路径总和 II](https://leetcode-cn.com/problems/path-sum-ii/)

**方法一：深度优先搜索**

思路及算法

我们可以采用深度优先搜索的方式，枚举每一条从根节点到叶子节点的路径。当我们遍历到叶子节点，且此时路径和恰为目标和时，我们就找到了一条满足条件的路径。


``` go
/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */
func pathSum(root *TreeNode, targetSum int) [][]int {
	path, res := []int{}, [][]int{}
	var dfs func(*TreeNode, int)
	dfs = func(node *TreeNode, sum int) {
		if node == nil {
			return
		}
		sum -= node.Val
		path = append(path, node.Val)
		defer func() { path = path[:len(path)-1] }()
		if sum == 0 && node.Left == nil && node.Right == nil {
			res = append(res, append([]int(nil), path...))
		}
		dfs(node.Left, sum)
		dfs(node.Right, sum)
		// path = path[:len(path)-1]
	}
	dfs(root, targetSum)
	return res
}
```




复杂度分析

- 时间复杂度：O(N^2)，其中 N 是树的节点数。在最坏情况下，树的上半部分为链状，下半部分为完全二叉树，并且从根节点到每一个叶子节点的路径都符合题目要求。此时，路径的数目为 O(N)，并且每一条路径的节点个数也为 O(N)，因此要将这些路径全部添加进答案中，时间复杂度为 O(N^2)。

- 空间复杂度：O(N)，其中 N 是树的节点数。空间复杂度主要取决于栈空间的开销，栈中的元素个数不会超过树的节点数。






## [41. 缺失的第一个正数](https://leetcode-cn.com/problems/first-missing-positive/) 


**方法一：置换**

除了打标记以外，我们还可以使用置换的方法，将给定的数组「恢复」成下面的形式：

如果数组中包含 x x∈[1,N]，那么恢复后，数组的第 x - 1 个元素为 x。

``` go
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



**方法二：哈希表**

![](https://assets.leetcode-cn.com/solution-static/41/41_fig1.png)


``` go
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



``` go
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






## [155. 最小栈](https://leetcode-cn.com/problems/min-stack/)

``` go
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

``` go
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







