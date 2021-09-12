
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

[补充题22. IP地址与整数的转换](https://mp.weixin.qq.com/s/u-RahFTB3JIqND41HqtotQ)

[补充题3. 求区间最小数乘区](https://mp.weixin.qq.com/s/ABNN4lJpvttulwWaUTgYZQ)

[补充题7. 木头切割问题](https://mp.weixin.qq.com/s/o-1VJO2TQZjC5ROmV7CReA)

[补充题24. 双栈排序](https://mp.weixin.qq.com/s/6mgvQ4PhN6psEwklZHsn6w)

[补充题21. 字符串相减](https://mp.weixin.qq.com/s/kCue4c0gnLSw0HosFl_t7w)

[补充题19. 判断一个点是否在三角形内](https://mp.weixin.qq.com/s/qnVUJq4lmnLsXJgyHCXngA)

[补充题12. 二叉树的下一个节点](https://mp.weixin.qq.com/s/ug9KoqbrVFMPBTqX-ZaKbA)

[补充题20. 立方根]()

[补充题11. 翻转URL字符串里的单词]()

[补充题17. 两个有序数组第k小的数]()

[补充题10. 36进制减法](https://mp.weixin.qq.com/s/_A2Ctn3kDa21NPlpF9y-hg)

[补充题13. 中文数字转阿拉伯数字]()

[补充题18. 反转双向链表]()



[字节跳动某面试官的压箱题——灯泡开关](https://mp.weixin.qq.com/s/GPQ3EqmBLU_kCeKn1Ggyvg)

[319. 灯泡开关](https://leetcode-cn.com/problems/bulb-switcher/)



------


[补充题4. 手撕快速排序 912. 排序数组](https://leetcode-cn.com/problems/sort-an-array/)

```go
func sortArray(nums []int) []int {
	quick_sort(nums, 0, len(nums)-1)
	return nums
}
func quick_sort(A []int, start, end int) {
	if start < end {
		piv_pos := random_partition(A, start, end)
		quick_sort(A, start, piv_pos-1)
		quick_sort(A, piv_pos+1, end)
	}
}
func partition(A []int, start, end int) int {
	piv, i := A[start], start+1
	for j := start + 1; j <= end; j++ {
		if A[j] < piv {
			A[i], A[j] = A[j], A[i]
			i++
		}
	}
	A[start], A[i-1] = A[i-1], A[start]
	return i - 1
}
func random_partition(A []int, start, end int) int {
	rand.Seed(time.Now().Unix())
	random := start + rand.Int()%(end-start+1)
	A[start], A[random] = A[random], A[start]
	return partition(A, start, end)
}
```


[补充题6. 手撕堆排序 912. 排序数组](https://leetcode-cn.com/problems/sort-an-array/)

```go
func sortArray(nums []int) []int {
	heap_sort(nums)
	return nums
}
func heap_sort(A []int) {
	heap_size := len(A)
	build_maxheap(A, heap_size)
	for i := heap_size - 1; i >= 0; i-- {
		A[0], A[i] = A[i], A[0]
		heap_size--
		max_heapify(A, 0, heap_size)
	}
}
func build_maxheap(A []int, heap_size int) {
	for i := heap_size >> 1; i >= 0; i-- {
		max_heapify(A, i, heap_size)
	}
}
func max_heapify(A []int, i, heap_size int) {
	lson, rson, largest := i<<1+1, i<<1+2, i
	for lson < heap_size && A[largest] < A[lson] {
		largest = lson
	}
	for rson < heap_size && A[largest] < A[rson] {
		largest = rson
	}
	if i != largest {
		A[i], A[largest] = A[largest], A[i]
		max_heapify(A, largest, heap_size)
	}
}
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
func sortArray(nums []int) []int {
	merge_sort(nums, 0, len(nums)-1)
	return nums
}
func merge_sort(A []int, start, end int) {
	if start < end {
		mid := start + (end-start)>>1
		merge_sort(A, start, mid)
		merge_sort(A, mid+1, end)
		merge(A, start, mid, end)
	}
}
func merge(A []int, start, mid, end int) {
	Arr := make([]int, end-start+1)
	p, q, k := start, mid+1, 0
	for i := start; i <= end; i++ {
		if p > mid {
			Arr[k] = A[q]
			q++
		} else if q > end {
			Arr[k] = A[p]
			p++
		} else if A[p] < A[q] {
			Arr[k] = A[p]
			p++
		} else {
			Arr[k] = A[q]
			q++
		}
		k++
	}
	for p := 0; p < k; p++ {
		A[start] = Arr[p]
		start++
	}
}
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