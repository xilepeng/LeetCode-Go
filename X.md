


## [3. 无重复字符的最长子串](https://leetcode-cn.com/problems/longest-substring-without-repeating-characters/)

**解法一 滑动窗口**
1. (a)bcabcbb
2. (ab)cabcbb
3. (abc)abcbb
4. (abca)bcbb 当前字符和首字符重复
5. ~~a~~(bca)bcbb 删除首字符（收缩窗口）
6. ~~a~~(bcab)cbb 继续向后扫描（扩展窗口）
7. ~~ab~~(cab)cbb


```go
func lengthOfLongestSubstring(s string) (res int) {
	hash := make(map[byte]int, len(s)) // 哈希集合记录每个字符出现次数
	for i, j := 0, 0; j < len(s); j++ {
		hash[s[j]]++                // 首次存入哈希
		for ; hash[s[j]] > 1; i++ { // 出现字符和首字符重复，i++跳过首字符(收缩窗口)
			hash[s[i]]--    // 哈希记录次数减1
		}
		if res < j-i+1 {
			res = j - i + 1 // 统计无重复字符的最长子串
		}
	}
	return res
}
```

## [206. 反转链表](https://leetcode-cn.com/problems/reverse-linked-list/) 

**方法一：迭代**

```go
/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
*/
func reverseList(head *ListNode) *ListNode {
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
```



```go
/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */
func reverseList(head *ListNode) *ListNode {
	if head == nil || head.Next == nil { // 最小子问题：无 / 只有一个节点
		return head
	}
	newHead := reverseList(head.Next) // 递：1->2->3->4->5->nil
	head.Next.Next = head             // 归：5->4   (1->2->3->  4->5->nil)
	head.Next = nil                   //    4->nil
	return newHead
}
```