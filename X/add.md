
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

[补充题6. 手撕堆排序 912. 排序数组](https://leetcode-cn.com/problems/sort-an-array/)

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