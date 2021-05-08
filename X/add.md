
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

[字节跳动某面试官的压箱题——灯泡开关](https://mp.weixin.qq.com/s/GPQ3EqmBLU_kCeKn1Ggyvg)

------


[补充题4. 手撕快速排序 912. 排序数组](https://leetcode-cn.com/problems/sort-an-array/)

思路和算法

快速排序的主要思想是通过划分将待排序的序列分成前后两部分，其中前一部分的数据都比后一部分的数据要小，
然后再递归调用函数对两部分的序列分别进行快速排序，以此使整个序列达到有序。

快排思路：
1. 确定分界点 x：q[l], q[r], q[(l+r)>>1], 随机
2. 调整区间：left <= x, right >= x
3. 递归处理左右两边

### 方法一：快速排序

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
		if a[j] <= x { //逆序交换
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

### 方法二：快速排序
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

### 方法三：基于随机选取主元的快速排序

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
		if a[j] <= x { //逆序交换
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

### 方法一：堆排序（大根堆）
思路和算法

堆排序的思想就是先将待排序的序列建成大根堆，使得每个父节点的元素大于等于它的子节点。此时整个序列最大值即为堆顶元素，我们将其与末尾元素交换，使末尾元素为最大值，然后再调整堆顶元素使得剩下的 n−1 个元素仍为大根堆，再重复执行以上操作我们即能得到一个有序的序列。


```go
func sortArray(nums []int) []int {
	heapSort(nums)
	return nums
}
func heapSort(a []int) {
	heapSize := len(a)
	buildMaxHeap(a, heapSize)
	for i := heapSize - 1; i >= 0; i-- {
		a[0], a[i] = a[i], a[0] //堆顶(最大值)交换到末尾,堆顶元素和堆底元素交换 
		heapSize--              //把剩余待排序元素整理成堆
		maxHeapify(a, 0, heapSize)
	}
}
func buildMaxHeap(a []int, heapSize int) { // O(n)
	for i := heapSize / 2; i >= 0; i-- { // heapSize / 2后面都是叶子节点，不需要向下调整
		maxHeapify(a, i, heapSize)
	}
}
func maxHeapify(a []int, i, heapSize int) { // O(nlogn) 大根堆，如果堆顶节点小于叶子，向下调整 
	l, r, largest := i*2+1, i*2+2, i
	if l < heapSize && a[largest] < a[l] { //左儿子存在且大于a[largest]
		largest = l
	}
	if r < heapSize && a[largest] < a[r] { //右儿子存在且大于a[largest]
		largest = r
	}
	if largest != i {				
		a[largest], a[i] = a[i], a[largest] //堆顶调整为最大值
		maxHeapify(a, largest, heapSize)    //递归处理
	}
}
```

复杂度分析

- 时间复杂度：O(nlogn)。初始化建堆的时间复杂度为 O(n)，建完堆以后需要进行 n−1 次调整，一次调整（即 maxHeapify） 的时间复杂度为 O(logn)，那么 n−1 次调整即需要 O(nlogn) 的时间复杂度。因此，总时间复杂度为 O(n+nlogn)=O(nlogn)。

- 空间复杂度：O(1)。只需要常数的空间存放若干变量。





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

### 方法一：归并排序

思路

归并排序利用了分治的思想来对序列进行排序。
对一个长为 nn 的待排序的序列，我们将其分解成两个长度为 n/2 的子序列。
每次先递归调用函数使两个子序列有序，然后我们再线性合并两个有序的子序列使整个序列有序。

1. 确定分解点: mid := (l + r) / 2
2. 递归排序左右两边
3. 归并


```go
func sortArray(nums []int) []int {
	n := len(nums)
	temp := make([]int, n)
	mergeSort(nums, temp, 0, n-1)
	return nums
}
func mergeSort(A, temp []int, start, end int) {
	if start < end {
		mid := start + (end-start)>>1
		mergeSort(A, temp, start, mid)
		mergeSort(A, temp, mid+1, end)
		merge(A, temp, start, mid, end)
	}
}
func merge(A, temp []int, start, mid, end int) {
	i, j, k := start, mid+1, 0
	for ; i <= mid && j <= end; k++ {
		if A[i] <= A[j] {
			temp[k] = A[i]
			i++
		} else {
			temp[k] = A[j]
			j++
		}
	}
	for ; i <= mid; i++ {
		temp[k] = A[i]
		k++
	}
	for ; j <= end; j++ {
		temp[k] = A[j]
		k++
	}
	copy(A[start:end+1], temp)
}
```


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
		} else if A[p] <= A[q] {
			Arr[k] = A[p]
			p++
		} else {
			Arr[k] = A[q]
			q++
		}
		k++
	}
	// copy(A[start:end+1], Arr)
	for p := 0; p < k; p++ {
		A[start] = Arr[p]
		start++
	}
}
```

```go

func sortArray(nums []int) []int {
	n := len(nums)
	mergeSort(nums, 0, n-1)
	return nums
}
func mergeSort(A []int, start, end int) {
	if start < end {
		mid := start + (end-start)>>1
		mergeSort(A, start, mid)
		mergeSort(A, mid+1, end)
		merge(A, start, mid, end)
	}
}
func merge(A []int, start, mid, end int) {
	temp := []int{}
	i, j := start, mid+1
	for k := start; k <= end; k++ {
		if i > mid {
			temp = append(temp, A[j])
			j++
		} else if j > end {
			temp = append(temp, A[i])
			i++
		} else if A[i] <= A[j] {
			temp = append(temp, A[i])
			i++
		} else {
			temp = append(temp, A[j])
			j++
		}
	}
	copy(A[start:end+1], temp)
}
```


### 方法二：归并排序

```go
func sortArray(nums []int) []int {
	mergeSort(nums, 0, len(nums)-1)
	return nums
}
func mergeSort(a []int, l, r int) {
	if l < r {
		mid := (l + r) >> 1
		mergeSort(a, l, mid)
		mergeSort(a, mid+1, r)
		merge(a, l, mid, r)
	}
}
func merge(a []int, l, mid, r int) {
	tmp := []int{}
	i, j := l, mid+1
	for i <= mid || j <= r {
		if i > mid || j <= r && a[j] < a[i] {
			tmp = append(tmp, a[j])
			j++
		} else {
			tmp = append(tmp, a[i])
			i++
		}
	}
	copy(a[l:r+1], tmp)
}
```

```go
func sortArray(nums []int) []int {
	mergeSort(nums, 0, len(nums)-1)
	return nums
}
func mergeSort(a []int, l, r int) {
	if l < r {
		mid := (l + r) >> 1
		mergeSort(a, l, mid)
		mergeSort(a, mid+1, r)
		tmp := []int{}
		i, j := l, mid+1
		for i <= mid || j <= r {
			if i > mid || j <= r && a[j] < a[i] {
				tmp = append(tmp, a[j])
				j++
			} else {
				tmp = append(tmp, a[i])
				i++
			}
		}
		copy(a[l:r+1], tmp)
	}
}
```


复杂度分析

- 时间复杂度：O(nlogn)。由于归并排序每次都将当前待排序的序列折半成两个子序列递归调用，然后再合并两个有序的子序列，而每次合并两个有序的子序列需要 O(n) 的时间复杂度，所以我们可以列出归并排序运行时间 T(n) 的递归表达式：

T(n)=2T(n/2)+O(n)

​ 根据主定理我们可以得出归并排序的时间复杂度为 O(nlogn)。

- 空间复杂度：O(n)。我们需要额外 O(n) 空间的 tmp 数组，且归并排序递归调用的层数最深为 log_2 n，所以我们还需要额外的 O(logn) 的栈空间，所需的空间复杂度即为 O(n+logn)=O(n)。



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