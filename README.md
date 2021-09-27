
**[CodeTop](https://codetop.cc/home)**


------


1. [✅ 206. 反转链表](#-206-反转链表)
1. [✅ 3. 无重复字符的最长子串](#-3-无重复字符的最长子串)
1. [✅ 146. LRU 缓存机制](#-146-lru-缓存机制)
1. [✅ 215. 数组中的第K个最大元素](#-215-数组中的第k个最大元素)
1. [✅ 25. K 个一组翻转链表](#-25-k-个一组翻转链表)
1. [✅ 补充题4. 手撕快速排序 912. 排序数组 ](#-补充题4-手撕快速排序-912-排序数组-)
1. [1. 两数之和](#1-两数之和)
1. [53. 最大子序和](#53-最大子序和)
1. [21. 合并两个有序链表](#21-合并两个有序链表)
1. [160. 相交链表](#160-相交链表)
1. [15. 三数之和](#15-三数之和)
1. [141. 环形链表](#141-环形链表)
1. [102. 二叉树的层序遍历](#102-二叉树的层序遍历)
1. [121. 买卖股票的最佳时机](#121-买卖股票的最佳时机)
1. [415. 字符串相加](#415-字符串相加)
1. [103. 二叉树的锯齿形层序遍历](#103-二叉树的锯齿形层序遍历)
1. [88. 合并两个有序数组](#88-合并两个有序数组)
1. [236. 二叉树的最近公共祖先](#236-二叉树的最近公共祖先)
1. [20. 有效的括号](#20-有效的括号)
1. [704. 二分查找](#704-二分查找)




------


## ✅ [206. 反转链表](https://leetcode-cn.com/problems/reverse-linked-list/) 


**方法一：迭代（双指针）**


![](images/206-1.jpg)

思路：在遍历链表时，将当前节点的 next 指针改为指向前一个节点。由于节点没有引用其前一个节点，因此必须事先存储其前一个节点。在更改引用之前，还需要存储后一个节点。最后返回新的头引用。


```go
/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */
func reverseList(head *ListNode) *ListNode {
	var prev *ListNode //prev -> nil
	curr := head
	for curr != nil {
		next := curr.Next //1.存储下一个节点
		curr.Next = prev  //2.当前节点的next指针指向前一个节点
		prev = curr       //3.移动双指针
		curr = next
	}
	return prev
}
```

复杂度分析

- 时间复杂度：O(n)，其中 n 是链表的长度。需要遍历链表一次。

- 空间复杂度：O(1)。



**方法二：头插法**

![](images/206-2.png)
![](images/206-2-1.png)

```go
/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */
func reverseList(head *ListNode) *ListNode {
    if head == nil || head.Next == nil {
        return head
    }
	dummy, curr := &ListNode{Next: head}, head
	for curr.Next != nil {
		next := curr.Next
		curr.Next = next.Next
		next.Next = dummy.Next //插入链表头
		dummy.Next = next
	}
	return dummy.Next
}
```


**方法三：递归**

![](images/206-3.png))

递归版本稍微复杂一些，其关键在于反向工作。假设链表的其余部分已经被反转，现在应该如何反转它前面的部分？

假设链表为：

n1 → … → nk−1 → nk → nk+1 → … → nm → ∅

若从节点 nk+1 到 nm 已经被反转，而我们正处于 nk。
n1 → … → nk−1 → nk → nk+1 ← … ← nm
​	
我们希望 nk+1 的下一个节点指向 nk。

所以，nk.next.next = nk 。
需要注意的是 n1 的下一个节点必须指向 ∅。如果忽略了这一点，链表中可能会产生环。



**思路**

首先我们先考虑 reverseList 函数能做什么，它可以翻转一个链表，并返回新链表的头节点，也就是原链表的尾节点。
所以我们可以先递归处理 reverseList(head->next)，这样我们可以将以 head->next 为头节点的链表翻转，并得到原链表的尾节 tail，此时 head->next 是新链表的尾节点，我们令它的 next 指针指向 head，并将 head->next 指向空即可将整个链表翻转，且新链表的头节点是tail。

- 空间复杂度分析：总共递归 n 层，系统栈的空间复杂度是 O(n)，所以总共需要额外 O(n) 的空间。
- 时间复杂度分析：链表中每个节点只被遍历一次，所以时间复杂度是 O(n)。


```go
func reverseList(head *ListNode) *ListNode {
	if head == nil || head.Next == nil { //只有一个节点或没有节点
		return head
	}
	newHead := reverseList(head.Next) //反转 head.Next
	head.Next.Next = head             //反转 head
	head.Next = nil
	return newHead
}
```

复杂度分析
- 时间复杂度：O(n)，其中 n 是链表的长度。需要对链表的每个节点进行反转操作。

- 空间复杂度：O(n)，其中 n 是链表的长度。空间复杂度主要取决于递归调用的栈空间，最多为 n 层。




## ✅ [3. 无重复字符的最长子串](https://leetcode-cn.com/problems/longest-substring-without-repeating-characters/)

![](images/3.png)

```go
func lengthOfLongestSubstring(s string) int {
	start, res := 0, 0
	m := map[byte]int{}
	for i := 0; i < len(s); i++ {
		if _, exists := m[s[i]]; exists {
			start = max(start, m[s[i]]+1)
		}
		m[s[i]] = i
		res = max(res, i-start+1)
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

T(n) = O(n)
S(n) = O(|Σ|) 其中 Σ 表示字符集（即字符串中可以出现的字符），∣Σ∣ 表示字符集的大小。

**方法二：(双指针扫描)**  O(n)

定义两个指针 i,j(i<=j)，表示当前扫描到的子串是 [i,j] (闭区间)。扫描过程中维护一个哈希表 m := map[byte]int{}，表示 [i,j] 中每个字符出现的次数。
线性扫描时，每次循环的流程如下：

指针 j
 向后移一位, 同时将哈希表中 s[j]
 的计数加一: m[s[j]]++;
假设 j
 移动前的区间 [i,j]
 中没有重复字符，则 j
 移动后，只有 s[j]
 可能出现2次。因此我们不断向后移动 i
，直至区间 [i,j]
中 s[j]
 的个数等于1为止；
复杂度分析：由于 i,j
 均最多增加n次，且哈希表的插入和更新操作的复杂度都是 O(1)
，因此，总时间复杂度 O(n)
.

```go
func lengthOfLongestSubstring(s string) int {
	m := map[byte]int{}
	res := 0
	for i, j := 0, 0; j < len(s); j++ {
		m[s[j]]++
		for m[s[j]] > 1 {
			m[s[i]]--
			i++
		}
		res = max(res, j-i+1)
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

```go
func lengthOfLongestSubstring(s string) int {
	m := map[byte]int{}
	res, start := 0, 0
	for i := 0; i < len(s); i++ {
		m[s[i]]++
		for m[s[i]] > 1 {
			m[s[start]]--
			start++
		}
		res = max(res, i-start+1)
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




复杂度分析

- 时间复杂度：O(N)，其中 N 是字符串的长度。左指针和右指针分别会遍历整个字符串一次。

- 空间复杂度：O(∣Σ∣)，其中 Σ 表示字符集（即字符串中可以出现的字符），∣Σ∣ 表示字符集的大小。在本题中没有明确说明字符集，因此可以默认为所有 ASCII 码在 [0, 128)[0,128) 内的字符，即∣Σ∣=128。我们需要用到哈希集合来存储出现过的字符，而字符最多有 ∣Σ∣ 个，因此空间复杂度为O(∣Σ∣)。


模拟：

 输入: s = "abcabcbb"
 输出: 3

 i = 0 m[a] = 1 res = 1 
 [a]bcabcbb

 i = 1 m[b] = 1 res = 1 - 0 + 1 = 2
 [ab]cabcbb
 
 i = 2 m[c] = 1 res = 2 - 0 + 1 = 3
 [abc]abcbb
 
 i = 3 m[a] = 2
 	{m[a] = 2-1 = 1 j = 1  res = 3 - 1 + 1 = 3}
 a[bca]bcbb





## ✅ [146. LRU 缓存机制](https://leetcode-cn.com/problems/lru-cache/)

![](images/146.png)

**方法一：哈希表 + 双向链表**
算法

LRU 缓存机制可以通过哈希表辅以双向链表实现，我们用一个哈希表和一个双向链表维护所有在缓存中的键值对。

- 双向链表按照被使用的顺序存储了这些键值对，靠近头部的键值对是最近使用的，而靠近尾部的键值对是最久未使用的。

- 哈希表即为普通的哈希映射（HashMap），通过缓存数据的键映射到其在双向链表中的位置。

这样以来，我们首先使用哈希表进行定位，找出缓存项在双向链表中的位置，随后将其移动到双向链表的头部，即可在 O(1)O(1) 的时间内完成 get 或者 put 操作。具体的方法如下：

- 对于 get 操作，首先判断 key 是否存在：

    如果 key 不存在，则返回 −1；

    如果 key 存在，则 key 对应的节点是最近被使用的节点。通过哈希表定位到该节点在双向链表中的位置，并将其移动到双向链表的头部，最后返回该节点的值。

- 对于 put 操作，首先判断 key 是否存在：

    如果 key 不存在，使用 key 和 value 创建一个新的节点，在双向链表的头部添加该节点，并将 key 和该节点添加进哈希表中。然后判断双向链表的节点数是否超出容量，如果超出容量，则删除双向链表的尾部节点，并删除哈希表中对应的项；

    如果 key 存在，则与 get 操作类似，先通过哈希表定位，再将对应的节点的值更新为 value，并将该节点移到双向链表的头部。

上述各项操作中，访问哈希表的时间复杂度为 (1)，在双向链表的头部添加节点、在双向链表的尾部删除节点的复杂度也为 O(1)。而将一个节点移到双向链表的头部，可以分成「删除该节点」和「在双向链表的头部添加节点」两步操作，都可以在 O(1) 时间内完成。

小贴士

在双向链表的实现中，使用一个伪头部（dummy head）和伪尾部（dummy tail）标记界限，这样在添加节点和删除节点的时候就不需要检查相邻的节点是否存在。

![截屏2021-05-19 12.46.17.png](http://ww1.sinaimg.cn/large/007daNw2ly1gqnn2m3y0hj318y0owq9l.jpg)



```go
type LRUCache struct {
	cache          map[int]*DLinkedNode
	head, tail     *DLinkedNode
	size, capacity int
}

type DLinkedNode struct {
	key, value int
	prev, next *DLinkedNode
}

func initDLinkedNode(key, value int) *DLinkedNode {
	return &DLinkedNode{
		key:   key,
		value: value,
	}
}

func Constructor(capacity int) LRUCache {
	l := LRUCache{
		cache:    map[int]*DLinkedNode{},
		head:     initDLinkedNode(0, 0),
		tail:     initDLinkedNode(0, 0),
		capacity: capacity,
	}
	l.head.next = l.tail
	l.tail.prev = l.head
	return l
}

func (this *LRUCache) Get(key int) int {
	if _, ok := this.cache[key]; !ok {
		return -1
	}
	node := this.cache[key] // 如果 key 存在，先通过哈希表定位，再移到头部
	this.moveToHead(node)
	return node.value
}

func (this *LRUCache) Put(key int, value int) {
	if _, ok := this.cache[key]; !ok { // 如果 key 不存在，创建一个新的节点
		node := initDLinkedNode(key, value)
		this.cache[key] = node // 添加进哈希表
		this.addToHead(node)   // 添加至双向链表的头部
		this.size++
		if this.size > this.capacity {
			removed := this.removeTail()    // 如果超出容量，删除双向链表的尾部节点
			delete(this.cache, removed.key) // 删除哈希表中对应的项
			this.size--
		}
	} else { // 如果 key 存在，先通过哈希表定位，再修改 value，并移到头部
		node := this.cache[key]
		node.value = value
		this.moveToHead(node)
	}
}

func (this *LRUCache) addToHead(node *DLinkedNode) {
	node.prev = this.head
	node.next = this.head.next
	this.head.next.prev = node
	this.head.next = node
}

func (this *LRUCache) removeNode(node *DLinkedNode) {
	node.prev.next = node.next
	node.next.prev = node.prev
}

func (this *LRUCache) moveToHead(node *DLinkedNode) {
	this.removeNode(node)
	this.addToHead(node)
}

func (this *LRUCache) removeTail() *DLinkedNode {
	node := this.tail.prev
	this.removeNode(node)
	return node
}

/**
 * Your LRUCache object will be instantiated and called as such:
 * obj := Constructor(capacity);
 * param_1 := obj.Get(key);
 * obj.Put(key,value);
 */
```

复杂度分析

时间复杂度：对于 put 和 get 都是 O(1)。

空间复杂度：O(capacity)，因为哈希表和双向链表最多存储 capacity+1 个元素。




## ✅ [215. 数组中的第K个最大元素](https://leetcode-cn.com/problems/kth-largest-element-in-an-array/)


* 考点1：能否实现解法的优化
* 考点2：是否了解快速选择算法
* 考点3：能否说明堆算法和快速选择算法的适用场景

**核心思路**

```go
func findKthLargest(nums []int, k int) int {
	n := len(nums)
	return quick_select(nums, 0, n-1, n-k)
}

func quick_select(A []int, start, end, i int) int {
	if piv_pos := partition(A, start, end); i == piv_pos {
		return A[i]
	} else if i < piv_pos {
		return quick_select(A, start, piv_pos-1, i)
	} else {
		return quick_select(A, piv_pos+1, end, i)
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
```


**方法一：基于快速排序的选择方法**

快速选择算法思路：

只要某次划分的 q 为倒数第 k 个下标的时候，我们就已经找到了答案。
如果划分得到的 q 正好就是我们需要的下标，就直接返回 a[q]；
否则，如果 q 比目标下标小，就递归右子区间，否则递归左子区间。

```go
func findKthLargest(nums []int, k int) int {
	rand.Seed(time.Now().UnixNano())
	n := len(nums)
	return quick_select(nums, 0, n-1, n-k)
}

func quick_select(A []int, start, end, i int) int {
	if piv_pos := random_partition(A, start, end); i == piv_pos {
		return A[i]
	} else if i < piv_pos {
		return quick_select(A, start, piv_pos-1, i)
	} else {
		return quick_select(A, piv_pos+1, end, i)
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
	random := rand.Int()%(end-start+1) + start
	A[start], A[random] = A[random], A[start]
	return partition(A, start, end)
}
```



复杂度分析

- 时间复杂度：O(n)，如上文所述，证明过程可以参考「《算法导论》9.2：期望为线性的选择算法」。
- 空间复杂度：O(logn)，递归使用栈空间的空间代价的期望为 O(logn)。



**方法二：基于堆排序的选择方法**

思路和算法

建立一个大根堆，做 k - 1 次删除操作后堆顶元素就是我们要找的答案。

```go
func findKthLargest(A []int, k int) int {
	heap_size, n := len(A), len(A)
	build_maxheap(A, heap_size)
	for i := heap_size - 1; i >= n-k+1; i-- {
		A[0], A[i] = A[i], A[0]
		heap_size--
		max_heapify(A, 0, heap_size)
	}
	return A[0]
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

复杂度分析

- 时间复杂度：O(nlogn)，建堆的时间代价是 O(n)，删除的总代价是 O(klogn)，因为 k < n，故渐进时间复杂为 O(n+klogn)=O(nlogn)。
- 空间复杂度：O(logn)，即递归使用栈空间的空间代价。





## ✅ [25. K 个一组翻转链表](https://leetcode-cn.com/problems/reverse-nodes-in-k-group/)

![截屏2021-04-21 11.31.37.png](http://ww1.sinaimg.cn/large/007daNw2ly1gpr7jmcf05j315g0pen05.jpg)


```go
/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */
func reverseKGroup(head *ListNode, k int) *ListNode {
	dummy := &ListNode{Next: head}
	prev := dummy
	for head != nil {
		tail := prev
		for i := 0; i < k; i++ {
			tail = tail.Next
			if tail == nil {
				return dummy.Next
			}
		}
		next := tail.Next
		tail.Next = nil
		prev.Next = reverse(head)
		head.Next = next
		prev = head
		head = next
	}
	return dummy.Next
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
```




复杂度分析

- 时间复杂度：O(n)，其中 n 为链表的长度。head 指针会在 O(⌊k/n⌋) 个节点上停留，每次停留需要进行一次 O(k) 的翻转操作。

- 空间复杂度：O(1)，我们只需要建立常数个变量。



## ✅ [补充题4. 手撕快速排序 912. 排序数组 ](https://leetcode-cn.com/problems/sort-an-array/)



**方法一：快速排序**

思路和算法

快速排序的主要思想是通过划分将待排序的序列分成前后两部分，其中前一部分的数据都比后一部分的数据要小，
然后再递归调用函数对两部分的序列分别进行快速排序，以此使整个序列达到有序。

快排思路：
1. 确定分界点 x：q[l], q[r], q[(l+r)>>1], 随机
2. 调整区间：left <= x, right >= x
3. 递归处理左右两边

- 时间复杂度： O(nlog(n)) 
- 空间复杂度： O(log(n)), 递归使用栈空间的空间代价为O(logn)。

```go
func sortArray(nums []int) []int {
	rand.Seed(time.Now().UnixNano())
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
	random := rand.Int()%(end-start+1) + start
	A[start], A[random] = A[random], A[start]
	return partition(A, start, end)
}
```

复杂度分析

- 时间复杂度：基于随机选取主元的快速排序时间复杂度为期望 O(nlogn)，其中 n 为数组的长度。详细证明过程可以见《算法导论》第七章，这里不再大篇幅赘述。

- 空间复杂度：O(h)，其中 h 为快速排序递归调用的层数。我们需要额外的 O(h) 的递归调用的栈空间，由于划分的结果不同导致了快速排序递归调用的层数也会不同，最坏情况下需 O(n) 的空间，最优情况下每次都平衡，此时整个递归树高度为 logn，空间复杂度为 O(logn)。








## [1. 两数之和](https://leetcode-cn.com/problems/two-sum/)


**方法一：暴力枚举**

思路及算法

最容易想到的方法是枚举数组中的每一个数 x，寻找数组中是否存在 target - x。

```go
func twoSum(nums []int, target int) []int {
	for i, x := range nums {
		for j := i + 1; j < len(nums); j++ {
			if x+nums[j] == target {
				return []int{i, j}
			}
		}
	}
	return nil
}
```
**方法二：哈希表**

思路及算法

使用哈希表，可以将寻找 target - x 的时间复杂度降低到从 O(N) 降低到 O(1)。

**查找表法**

- 在遍历的同时，记录一些信息，以省去一层循环，这是以空间换时间的想法
- 需要记录已经遍历过的数值和他所对应的下标，可以借鉴查找表实现
- 查找表有2个常用的实现：
1. 哈希表
2. 平衡二叉搜索树

![](http://ww1.sinaimg.cn/large/007daNw2ly1goda8rlev9j31qu0vs42j.jpg)



```go
func twoSum(nums []int, target int) []int {
	hashTable := map[int]int{}
	for i, x := range nums {
		if p, ok := hashTable[target-x]; ok {
			return []int{p, i}
		}
		hashTable[x] = i
	}
	return nil
}
```


*Data Structure:*
- HashMap:<num, the index of the num>

**Algorithm:**

从头开始遍历数组：
1. 在map里找到当前这个数的另一半，返回
2. 没找到，存入map, key为数， value为index

- Solution:先找到另一半，没有就存入
- Why HashMap ?

存储 num 和 index 的关系，便于快速查找

Any Detial?
- 找到答案就break结束
- 先查找另一半，再存入，避免和自己相加



## [53. 最大子序和](https://leetcode-cn.com/problems/maximum-subarray/)

**方法一：贪心**

- 若当前指针所指元素之前的和小于0， 则丢弃当前元素之前的数列
- 将当前值与最大值比较，取最大

![截屏2021-03-12 15.20.16.png](http://ww1.sinaimg.cn/large/007daNw2ly1goh5cv919kj30vk0e4wfn.jpg)

```go
func maxSubArray(nums []int) int {
	curSum, maxSum := nums[0], nums[0]
	for i := 1; i < len(nums); i++ {
		curSum = max(nums[i], curSum+nums[i])
		maxSum = max(maxSum, curSum)
	}
	return maxSum
}
func max(x, y int) int {
	if x > y {
		return x
	}
	return y
}
```

**方法二：动态规划**

- 若前一个元素大于0，将其加到当前元素上

![截屏2021-03-12 16.53.25.png](http://ww1.sinaimg.cn/large/007daNw2ly1goh9sg1mb9j31780dw3ze.jpg)

```go
func maxSubArray(nums []int) int {
	max := nums[0]
	for i := 1; i < len(nums); i++ {
		if nums[i-1] > 0 {
			nums[i] += nums[i-1]
		}
		if max < nums[i] {
			max = nums[i]
		}
	}
	return max
}
```







## [21. 合并两个有序链表](https://leetcode-cn.com/problems/merge-two-sorted-lists/)



```go
/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */
func mergeTwoLists(l1 *ListNode, l2 *ListNode) *ListNode {
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


```go
/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */
func mergeTwoLists(l1 *ListNode, l2 *ListNode) *ListNode {
	if l1 == nil {
		return l2
	}
	if l2 == nil {
		return l1
	}
	if l1.Val < l2.Val {
		l1.Next = mergeTwoLists(l1.Next, l2)
		return l1
	} else {
		l2.Next = mergeTwoLists(l1, l2.Next)
		return l2
	}
}
```


## [160. 相交链表](https://leetcode-cn.com/problems/intersection-of-two-linked-lists/)

**方法一：双指针法**

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













## [15. 三数之和](https://leetcode-cn.com/problems/3sum/)

**思路**

外层循环：指针 i 遍历数组。
内层循环：用双指针，去寻找满足三数之和 == 0 的元素

**先排序的意义**
便于跳过重复元素，如果当前元素和前一个元素相同，跳过。

**双指针的移动时，避免出现重复解**

找到一个解后，左右指针同时向内收缩，为了避免指向重复的元素，需要：

- 左指针在保证left < right的前提下，一直右移，直到指向不重复的元素
- 右指针在保证left < right的前提下，一直左移，直到指向不重复的元素

**小优化**

排序后，如果外层遍历的数已经大于0，则另外两个数一定大于0，sum不会等于0，直接break。



```go
func threeSum(nums []int) [][]int {
	sort.Ints(nums)
	res := [][]int{}
	for i := 0; i < len(nums)-2; i++ {
		n1 := nums[i]
		if n1 > 0 { //如果最小的数大于0，break
			break
		}
		if i > 0 && n1 == nums[i-1] { //如果和前一个相同，跳过
			continue
		} //转换为两数之和，双指针解法
		l, r := i+1, len(nums)-1
		for l < r {
			n2, n3 := nums[l], nums[r]
			if n1+n2+n3 == 0 {
				res = append(res, []int{n1, n2, n3})
				for l < r && nums[l] == n2 { //去重移位
					l++
				}
				for l < r && nums[r] == n3 {
					r--
				}
			} else if n1+n2+n3 < 0 {
				l++
			} else {
				r--
			}
		}
	}
	return res
}
```

复杂度分析

- 时间复杂度：O(N^2)，其中 N 是数组 nums 的长度。

- 空间复杂度：O(logN)。我们忽略存储答案的空间，额外的排序的空间复杂度为 O(logN)。然而我们修改了输入的数组 nums，在实际情况下不一定允许，因此也可以看成使用了一个额外的数组存储了 nums 的副本并进行排序，空间复杂度为 O(N)。





## [141. 环形链表](https://leetcode-cn.com/problems/linked-list-cycle/)

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









## [102. 二叉树的层序遍历](https://leetcode-cn.com/problems/binary-tree-level-order-traversal/)

**方法一：DFS递归**

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

**方法二：BFS(queue)迭代**

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







## [121. 买卖股票的最佳时机](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock/)

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






## [415. 字符串相加](https://leetcode-cn.com/problems/add-strings/)

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







## [103. 二叉树的锯齿形层序遍历](https://leetcode-cn.com/problems/binary-tree-zigzag-level-order-traversal/)

**方法一：深度优先遍历**

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

**方法二：广度优先遍历**

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



## [88. 合并两个有序数组](https://leetcode-cn.com/problems/merge-sorted-array/)

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


## [236. 二叉树的最近公共祖先](https://leetcode-cn.com/problems/lowest-common-ancestor-of-a-binary-tree/)

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


## [20. 有效的括号](https://leetcode-cn.com/problems/valid-parentheses/)

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



## [704. 二分查找](https://leetcode-cn.com/problems/binary-search/)

1. 如果目标值等于中间元素，则找到目标值。
2. 如果目标值较小，继续在左侧搜索。
3. 如果目标值较大，则继续在右侧搜索。

**算法：**

- 初始化指针 left = 0, right = n - 1。
- 当 left <= right：
比较中间元素 nums[mid] 和目标值 target 。
	1. 如果 target = nums[mid]，返回 mid。
	2. 如果 target < nums[mid]，则在左侧继续搜索 right = mid - 1。
	3. 如果 target > nums[mid]，则在右侧继续搜索 left = mid + 1。


```go
func search(nums []int, target int) int {
	left, right := 0, len(nums)-1
	for left <= right {
		mid := left + (right-left)>>1
		if nums[mid] == target {
			return mid
		} else if nums[mid] < target {
			left = mid + 1
		} else {
			right = mid - 1
		}
	}
	return -1
}
```
复杂度分析

- 时间复杂度：O(logN)。
- 空间复杂度：O(1)。


















------
<!-- 

[206. 反转链表](https://leetcode-cn.com/problems/reverse-linked-list/) 

[3. 无重复字符的最长子串](https://leetcode-cn.com/problems/longest-substring-without-repeating-characters/)

[215. 数组中的第K个最大元素](https://leetcode-cn.com/problems/kth-largest-element-in-an-array/)


[146. LRU 缓存机制](https://leetcode-cn.com/problems/lru-cache/)

[补充题4. 手撕快速排序 912. 排序数组](https://leetcode-cn.com/problems/sort-an-array/)

[25. K 个一组翻转链表](https://leetcode-cn.com/problems/reverse-nodes-in-k-group/)

[1. 两数之和](https://leetcode-cn.com/problems/two-sum/)

[53. 最大子序和](https://leetcode-cn.com/problems/maximum-subarray/)

[21. 合并两个有序链表](https://leetcode-cn.com/problems/merge-two-sorted-lists/)

[160. 相交链表](https://leetcode-cn.com/problems/intersection-of-two-linked-lists/)



[15. 三数之和](https://leetcode-cn.com/problems/3sum/)

[141. 环形链表](https://leetcode-cn.com/problems/linked-list-cycle/)

[102. 二叉树的层序遍历](https://leetcode-cn.com/problems/binary-tree-level-order-traversal/)

[121. 买卖股票的最佳时机](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock/)

[415. 字符串相加](https://leetcode-cn.com/problems/add-strings/)

[103. 二叉树的锯齿形层序遍历](https://leetcode-cn.com/problems/binary-tree-zigzag-level-order-traversal/)

[88. 合并两个有序数组](https://leetcode-cn.com/problems/merge-sorted-array/)

[236. 二叉树的最近公共祖先](https://leetcode-cn.com/problems/lowest-common-ancestor-of-a-binary-tree/)

[20. 有效的括号](https://leetcode-cn.com/problems/valid-parentheses/)

[704. 二分查找](https://leetcode-cn.com/problems/binary-search/) -->



