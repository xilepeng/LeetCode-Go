
[206. 反转链表](https://leetcode-cn.com/problems/reverse-linked-list/) 

[215. 数组中的第K个最大元素](https://leetcode-cn.com/problems/kth-largest-element-in-an-array/)

[3. 无重复字符的最长子串](https://leetcode-cn.com/problems/longest-substring-without-repeating-characters/)

[146. LRU 缓存机制](https://leetcode-cn.com/problems/lru-cache/)

[25. K 个一组翻转链表](https://leetcode-cn.com/problems/reverse-nodes-in-k-group/)

[补充题4. 手撕快速排序 912. 排序数组](https://leetcode-cn.com/problems/sort-an-array/)

[1. 两数之和](https://leetcode-cn.com/problems/two-sum/)

[15. 三数之和](https://leetcode-cn.com/problems/3sum/)

[21. 合并两个有序链表](https://leetcode-cn.com/problems/merge-two-sorted-lists/)

[53. 最大子序和](https://leetcode-cn.com/problems/maximum-subarray/)


------



[206. 反转链表](https://leetcode-cn.com/problems/reverse-linked-list/) 


### 方法一：双指针（迭代）

![截屏2021-04-13 14.14.02.png](http://ww1.sinaimg.cn/large/007daNw2ly1gpi39qgmnoj311w0o6n2k.jpg)


思路：将当前节点的 next 指针改为指向前一个节点。

```go
func reverseList(head *ListNode) *ListNode {
	var prev *ListNode
	curr := head
	for curr != nil {
		next := curr.Next //1.存储下一个节点
		curr.Next = prev  //2.反转
		prev = curr       //3.移动双指针
		curr = next
	}
	return prev
}
```

复杂度分析

- 时间复杂度：O(n)，其中 n 是链表的长度。需要遍历链表一次。

- 空间复杂度：O(1)。



### 方法二：头插法

![截屏2021-04-21 12.11.01.png](http://ww1.sinaimg.cn/large/007daNw2ly1gpr8ocoatoj316s0o844c.jpg)

![截屏2021-04-21 12.11.12.png](http://ww1.sinaimg.cn/large/007daNw2ly1gpr8otljdfj315c0oy42w.jpg)


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


### 方法三：递归

![截屏2021-04-13 14.08.51.png](http://ww1.sinaimg.cn/large/007daNw2ly1gpi34y26nwj312u0lejug.jpg)

递归版本稍微复杂一些，其关键在于反向工作。假设链表的其余部分已经被反转，现在应该如何反转它前面的部分？

假设链表为：

n1 → … → nk−1 → nk → nk+1 → … → nm → ∅

若从节点 nk+1 到 nm 已经被反转，而我们正处于 nk。
n1 → … → nk−1 → nk → nk+1 ← … ← nm
​	
我们希望 nk+1 的下一个节点指向 nk。

所以，nk.next.next = nk 。
需要注意的是 n1 的下一个节点必须指向 ∅。如果忽略了这一点，链表中可能会产生环。



#### 思路

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



[215. 数组中的第K个最大元素](https://leetcode-cn.com/problems/kth-largest-element-in-an-array/)


* 考点1：能否实现解法的优化
* 考点2：是否了解快速选择算法
* 考点3：能否说明堆算法和快速选择算法的适用场景

## 方法一：基于快速排序的选择方法

快速选择算法思路：

1. 随机确定分界点 p
2. 调整区间：pIdx <= index, 递归右子区间; pIdx <= index, 递归右边
3. 递归处理左边或右边 
只要某次划分的 q 为倒数第 k 个下标的时候，我们就已经找到了答案。
如果划分得到的 q 正好就是我们需要的下标，就直接返回 a[q]；
否则，如果 q 比目标下标小，就递归右子区间，否则递归左子区间。

时间复杂度： O(n) 
空间复杂度： O(log(n)), 递归使用栈空间的空间代价为O(logn)。

```go
func findKthLargest(nums []int, k int) int {
    rand.Seed(time.Now().UnixNano())
	return quickSelect(nums, 0, len(nums)-1, len(nums)-k)
}
func quickSelect(a []int, l, r, index int) int {
	p := randomPartition(a, l, r)
	if p == index {
		return a[p]
	} else if p < index {
		return quickSelect(a, p+1, r, index)
	} 
	return quickSelect(a, l, p-1, index)
}

func randomPartition(a[]int, l, r int) int {
    i := rand.Int() % (r - l + 1) + l 
    a[r], a[i] = a[i], a[r]
    return partition(a, l, r)
}
func partition(a []int, l, r int) int {
	i := l - 1
	for j := l; j < r; j++ {
		if a[j] <= a[r] {
			i++
			a[i], a[j] = a[j], a[i]
		}
	}
	a[i+1], a[r] = a[r], a[i+1]
	return i + 1
}
```


## 方法二：基于堆排序的选择方法

思路和算法

建立一个大根堆，做 k - 1 次删除操作后堆顶元素就是我们要找的答案。

```go
func findKthLargest(nums []int, k int) int {
    heapSize := len(nums)
    buildMaxHeap(nums, heapSize)
    for i := len(nums) - 1; i >= len(nums) - k + 1; i-- {
        nums[0], nums[i] = nums[i], nums[0]
        heapSize--
        maxHeapify(nums, 0, heapSize)
    }
    return nums[0]
}

func buildMaxHeap(a []int, heapSize int) {
    for i := heapSize/2; i >= 0; i-- {
        maxHeapify(a, i, heapSize)
    }
}

func maxHeapify(a []int, i, heapSize int) {
    l, r, largest := i * 2 + 1, i * 2 + 2, i
    if l < heapSize && a[l] > a[largest] {
        largest = l
    }
    if r < heapSize && a[r] > a[largest] {
        largest = r
    }
    if largest != i {
        a[i], a[largest] = a[largest], a[i]
        maxHeapify(a, largest, heapSize)
    }
}

```

复杂度分析

- 时间复杂度：O(nlogn)，建堆的时间代价是 O(n)，删除的总代价是 O(klogn)，因为 k < n，故渐进时间复杂为 O(n+klogn)=O(nlogn)。
- 空间复杂度：O(logn)，即递归使用栈空间的空间代价。




[3. 无重复字符的最长子串](https://leetcode-cn.com/problems/longest-substring-without-repeating-characters/)

```go
func lengthOfLongestSubstring(s string) int {
	m := map[byte]int{}
	res := 0
	for i, j := 0, 0; i < len(s); i++ {
		m[s[i]]++
		for m[s[i]] > 1 {
			m[s[j]]--
			j++
		}
		res = max(res, i-j+1)
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



[146. LRU 缓存机制](https://leetcode-cn.com/problems/lru-cache/)

* 考点1：实现哈希表解法
* 考点2：想到利用链表解决访问顺序问题 

![截屏2021-03-03 21.39.46.png](http://ww1.sinaimg.cn/large/007daNw2ly1go71rbbeg4j31qy0z8gui.jpg)

```go
type LRUCache struct {
	size       int
	capacity   int
	cache      map[int]*DLinkedNode
	head, tail *DLinkedNode
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
	node := this.cache[key]
	this.moveToHead(node)
	return node.value
}

func (this *LRUCache) Put(key int, value int) {
	if _, ok := this.cache[key]; !ok {
		node := initDLinkedNode(key, value)
		this.cache[key] = node
		this.addToHead(node)
		this.size++
		if this.size > this.capacity {
			removed := this.removeTail()
			delete(this.cache, removed.key)
			this.size--
		}
	} else {
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
```

[25. K 个一组翻转链表](https://leetcode-cn.com/problems/reverse-nodes-in-k-group/)

![截屏2021-04-21 11.31.37.png](http://ww1.sinaimg.cn/large/007daNw2ly1gpr7jmcf05j315g0pen05.jpg)

```go
func reverseKGroup(head *ListNode, k int) *ListNode {
	dummy := &ListNode{Next: head}
	prev := dummy
	for head != nil {
		for i := 0; i < k-1 && head != nil; i++ {
			head = head.Next
		}
		if head == nil {
			break
		}
		curr := prev.Next
		next := head.Next
		head.Next = nil
		prev.Next = reverse(curr)
		curr.Next = next
		head = next
		prev = curr
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




[补充题4. 手撕快速排序 912. 排序数组 ](https://leetcode-cn.com/problems/sort-an-array/)

* 考点1：能否实现解法的优化
* 考点2：是否了解快速选择算法
* 考点3：能否说明堆算法和快速选择算法的适用场景

### 方法一：快速排序

思路和算法

快速排序的主要思想是通过划分将待排序的序列分成前后两部分，其中前一部分的数据都比后一部分的数据要小，
然后再递归调用函数对两部分的序列分别进行快速排序，以此使整个序列达到有序。

快排思路：
1. 确定分界点 x：q[l], q[r], q[(l+r)>>1], 随机
2. 调整区间：left <= x, right >= x
3. 递归处理左右两边

时间复杂度： O(nlog(n)) 
空间复杂度： O(log(n)), 递归使用栈空间的空间代价为O(logn)。

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
	x, i := a[r], l-1
	for j := l; j < r; j++ {
		if a[j] < x {
			i++
			a[i], a[j] = a[j], a[i] //逆序 交换
		}
	}
	a[i+1], a[r] = a[r], a[i+1]
	return i + 1
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
	a[i], a[r] = a[r], a[i]
	return partition(a, l, r)
}
func partition(a []int, l, r int) int {
	x, i := a[r], l-1
	for j := l; j < r; j++ {
		if a[j] < x {
			i++
			a[i], a[j] = a[j], a[i] //逆序 交换
		}
	}
	a[i+1], a[r] = a[r], a[i+1]
	return i + 1
}
```


[1. 两数之和](https://leetcode-cn.com/problems/two-sum/)


方法一：暴力枚举
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
方法二：哈希表

思路及算法

使用哈希表，可以将寻找 target - x 的时间复杂度降低到从 O(N) 降低到 O(1)。

## 查找表法

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

## Algorithm:
从头开始遍历数组：
1. 在map里找到当前这个数的另一半，返回
2. 没找到，存入map, key为数， value为index

- Solution:先找到另一半，没有就存入
- Why HashMap ?

存储 num 和 index 的关系，便于快速查找

Any Detial?
- 找到答案就break结束
- 先查找另一半，再存入，避免和自己相加



[15. 三数之和](https://leetcode-cn.com/problems/3sum/)

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


[21. 合并两个有序链表](https://leetcode-cn.com/problems/merge-two-sorted-lists/)



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





[53. 最大子序和](https://leetcode-cn.com/problems/maximum-subarray/)

方法一：贪心

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

方法二：动态规划

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





















