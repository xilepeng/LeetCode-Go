
[补充题4. 手撕快速排序 912. 排序数组](https://leetcode-cn.com/problems/sort-an-array/)

[补充题6. 手撕堆排序 912. 排序数组](https://leetcode-cn.com/problems/sort-an-array/)

[补充题1. 排序奇升偶降链表](https://mp.weixin.qq.com/s/377FfqvpY8NwMInhpoDgsw)

[补充题5. 手撕归并排序 912. 排序数组](https://leetcode-cn.com/problems/sort-an-array/)

[补充题23. 检测循环依赖](https://mp.weixin.qq.com/s/q6AhBt6MX2RL_HNZc8cYKQ)

[补充题2. 圆环回原点问题](https://mp.weixin.qq.com/s/VnGFEWHeD3nh1n9JSDkVUg)

[补充题9. 36进制加法](https://mp.weixin.qq.com/s/bgD1Q5lc92mX7RNS1L65qA)

[415. 字符串相加](https://leetcode-cn.com/problems/add-strings/)

[补充题14. 阿拉伯数字转中文数字]()

[补充题8. 计算数组的小和](https://mp.weixin.qq.com/s/0ih4W6nawzFUPSj3GOnYTQ)

[补充题3. 求区间最小数乘区](https://mp.weixin.qq.com/s/ABNN4lJpvttulwWaUTgYZQ)

[补充题7. 木头切割问题](https://mp.weixin.qq.com/s/o-1VJO2TQZjC5ROmV7CReA)

[补充题21. 字符串相减](https://mp.weixin.qq.com/s/kCue4c0gnLSw0HosFl_t7w)

[补充题19. 判断一个点是否在三角形内](https://mp.weixin.qq.com/s/qnVUJq4lmnLsXJgyHCXngA)

[补充题12. 二叉树的下一个节点](https://mp.weixin.qq.com/s/ug9KoqbrVFMPBTqX-ZaKbA)

[补充题11. 翻转URL字符串里的单词]()

[补充题17. 两个有序数组第k小的数]()

[补充题10. 36进制减法](https://mp.weixin.qq.com/s/_A2Ctn3kDa21NPlpF9y-hg)

[补充题13. 中文数字转阿拉伯数字]()

[补充题18. 反转双向链表]()

[补充题20. 立方根]()

------


[补充题4. 手撕快速排序 912. 排序数组](https://leetcode-cn.com/problems/sort-an-array/)

思路和算法

快速排序的主要思想是通过划分将待排序的序列分成前后两部分，其中前一部分的数据都比后一部分的数据要小，
然后再递归调用函数对两部分的序列分别进行快速排序，以此使整个序列达到有序。

快排思路：
1. 确定分界点 x：q[l], q[r], q[(l+r)>>1], 随机
2. 调整区间：left <= x, right >= x
3. 递归处理左右两边



```go
func sortArray(nums []int) []int {
	quickSort(nums, 0, len(nums)-1)
	return nums
}
func quickSort(a []int, l, r int) {
	if l < r {
		pos := partition(a, l, r)
		quickSort(a, l, pos-1)
		quickSort(a, pos+1, r)
	}
}
func partition(a []int, l, r int) int {
	a[r], a[(l+r)>>1] = a[(l+r)>>1], a[r]
	x, i := a[r], l-1
	for j := l; j < r; j++ {
		if a[j] <= x { //逆序 交换
			i++
			a[i], a[j] = a[j], a[i]
		}
	}
	a[i+1], a[r] = a[r], a[i+1]
	return i + 1
}
```

- 时间复杂度： O(nlog(n)) 
- 空间复杂度： O(log(n)), 递归使用栈空间的空间代价为O(logn)。


```go
func sortArray(nums []int) []int {
	quickSort(nums, 0, len(nums)-1)
	return nums
}
func quickSort(a []int, l, r int) {
	if l < r {
		a[(l+r)>>1], a[r] = a[r], a[(l+r)>>1]
		i := l - 1
		for j := l; j < r; j++ {
			if a[j] <= a[r] { //逆序交换
				i++
				a[i], a[j] = a[j], a[i]
			}
		}
		i++
		a[i], a[r] = a[r], a[i]
		quickSort(a, l, i-1)
		quickSort(a, i+1, r)
	}
}
```

```go
func sortArray(nums []int) []int {
	rand.Seed(time.Now().UnixNano())
	quickSort(nums, 0, len(nums)-1)
	return nums
}
func quickSort(a []int, l, r int) {
	if l < r {
		pos := randomPartition(a, l, r)
		quickSort(a, l, pos-1)
		quickSort(a, pos+1, r)
	}
}
func randomPartition(a []int, l, r int) int {
	i := rand.Int()%(r-l+1) + l
	a[r], a[i] = a[i], a[r]
	return partition(a, l, r)
}
func partition(a []int, l, r int) int {
	x, i := a[r], l-1
	for j := l; j < r; j++ {
		if a[j] <= x { //逆序 交换
			i++
			a[i], a[j] = a[j], a[i]
		}
	}
	a[i+1], a[r] = a[r], a[i+1]
	return i + 1
}

```

复杂度分析

- 时间复杂度：基于随机选取主元的快速排序时间复杂度为期望 O(nlogn)，其中 n 为数组的长度。详细证明过程可以见《算法导论》第七章，这里不再大篇幅赘述。

- 空间复杂度：O(h)，其中 h 为快速排序递归调用的层数。我们需要额外的 O(h) 的递归调用的栈空间，由于划分的结果不同导致了快速排序递归调用的层数也会不同，最坏情况下需 O(n) 的空间，最优情况下每次都平衡，此时整个递归树高度为 logn，空间复杂度为 O(logn)。

[补充题6. 手撕堆排序 912. 排序数组](https://leetcode-cn.com/problems/sort-an-array/)


```go

```

[补充题1. 排序奇升偶降链表](https://mp.weixin.qq.com/s/377FfqvpY8NwMInhpoDgsw)

```go
/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */
func sortOddEvenList(head *ListNode) *ListNode {
	if head == nil || head.Next == nil {
		return head
	}
	odd, even := oddEven(head)
	even = reverse(even)
	return merge(odd, even)
}
func oddEven(head *ListNode) (*ListNode, *ListNode) {
	evenHead := head.Next
	odd, even := head, evenHead
	for even != nil && even.Next != nil {
		odd.Next = even.Next
		odd = odd.Next
		even.Next = odd.Next
		even = even.Next
	}
	return odd, even
}
func reverse(head *ListNode) *ListNode {
	var prev *ListNode
	curr := head
	for curr != nil {
		next := curr.Next
		curr.Next = prev
		prev = curr
		curr = next
	}
	return prev
}
func merge(l1, l2 *ListNode) *ListNode {
	dummy := new(ListNode)
	prev := dummy
	for l1 != nil && l2 != nil {
		if l1.Val < l2.Val {
			prev.Next = l1
			l1 = l1.Next
		} else {
			prev.Next = l2
			l2 = l2.Next
		}
		prev = prev.Next
	}
	if l1 != nil {
		prev.Next = l1
	} else {
		prev.Next = l2
	}
	return dummy.Next
}
```

[补充题5. 手撕归并排序 912. 排序数组](https://leetcode-cn.com/problems/sort-an-array/)

```go

```

[补充题23. 检测循环依赖](https://mp.weixin.qq.com/s/q6AhBt6MX2RL_HNZc8cYKQ)

[补充题2. 圆环回原点问题](https://mp.weixin.qq.com/s/VnGFEWHeD3nh1n9JSDkVUg)

[补充题9. 36进制加法](https://mp.weixin.qq.com/s/bgD1Q5lc92mX7RNS1L65qA)

[415. 字符串相加](https://leetcode-cn.com/problems/add-strings/)

[补充题14. 阿拉伯数字转中文数字]()

[补充题8. 计算数组的小和](https://mp.weixin.qq.com/s/0ih4W6nawzFUPSj3GOnYTQ)

[补充题3. 求区间最小数乘区](https://mp.weixin.qq.com/s/ABNN4lJpvttulwWaUTgYZQ)

[补充题7. 木头切割问题](https://mp.weixin.qq.com/s/o-1VJO2TQZjC5ROmV7CReA)

[补充题21. 字符串相减](https://mp.weixin.qq.com/s/kCue4c0gnLSw0HosFl_t7w)

[补充题19. 判断一个点是否在三角形内](https://mp.weixin.qq.com/s/qnVUJq4lmnLsXJgyHCXngA)

[补充题12. 二叉树的下一个节点](https://mp.weixin.qq.com/s/ug9KoqbrVFMPBTqX-ZaKbA)

[补充题11. 翻转URL字符串里的单词]()

[补充题17. 两个有序数组第k小的数]()

[补充题10. 36进制减法](https://mp.weixin.qq.com/s/_A2Ctn3kDa21NPlpF9y-hg)

[补充题13. 中文数字转阿拉伯数字]()

[补充题18. 反转双向链表]()

[补充题20. 立方根]()