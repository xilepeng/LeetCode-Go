

1. [手撕快速排序 ](#补充题4-手撕快速排序-)
2. [手撕归并排序](#补充题5-手撕归并排序)
3. [手撕堆排序 ](#补充题6-手撕堆排序-)
4. [插入排序](#插入排序)
5. [希尔排序](#希尔排序)
6. [选择排序](#选择排序)
7. [冒泡排序](#冒泡排序)
8. [Counting Sort](#counting-sort)
9. [Bucket Sort](#bucket-sort)
10. [Radix Sort](#radix-sort)




## [手撕快速排序 ](https://leetcode-cn.com/problems/sort-an-array/)

``` go
func sortArray(nums []int) []int {
	QuickSort(nums, 0, len(nums)-1)
	return nums
}

func QuickSort(A []int, Left, Right int) {
	if Left >= Right {
		return
	}
	Pivot := A[Left+((Right-Left)>>1)]
	i, j := Left-1, Right+1
	for i < j {
		for i++; A[i] < Pivot; i++ {
		}
		for j--; A[j] > Pivot; j-- {
		}
		if i < j {
			A[i], A[j] = A[j], A[i]
		}
	}
	QuickSort(A, Left, j)
	QuickSort(A, j+1, Right)
}
```

**1.优雅写法**

```go
func sortArray(A []int) []int {
	quickSort(A, 0, len(A)-1)
	return A
}

func quickSort(A []int, low, high int) {
	if low >= high {
		return
	}
	pos := partition(A, low, high)
	quickSort(A, low, pos-1)
	quickSort(A, pos+1, high)
}

func partition(A []int, low, high int) int {
	A[high], A[low+(high-low)>>1] = A[low+(high-low)>>1], A[high] // 以 A[high] 作为基准数
	i, j := low, high
	for i < j {
		for i < j && A[i] <= A[high] { // 从左向右找首个大于基准数的元素
			i++
		}
		for i < j && A[j] >= A[high] { // 从右向左找首个小于基准数的元素
			j--
		}
		A[i], A[j] = A[j], A[i] // 元素交换到正确的区间
	}
	A[i], A[high] = A[high], A[i] // 将基准数交换至两子数组的分界线
	return i                      // 返回基准数的索引
}

// 快速排序（尾递归优化）
func quickSort(A []int, low, high int) {
	for low < high { // 子数组长度为 1 时终止
		pos := partition(A, low, high) // 哨兵划分操作
		// 对两个子数组中较短的那个执行快排
		if pos-low < high-pos {
			quickSort(A, low, pos-1) // 递归排序左子数组
			low = pos + 1            // 剩余待排序区间为 [pos + 1, high]
		} else {
			quickSort(A, pos+1, high) // 递归排序右子数组
			high = pos - 1            // 剩余待排序区间为 [low, pos - 1]
		}
	}
}
```


*2.更快写法**

```go
func sortArray(A []int) []int {
	quickSort(A, 0, len(A)-1)
	return A
}

// 快速排序 O(nlogn)
// 基于分而治之的方法，随机选择枢轴元素划分数组，
// 左边小于枢轴、右边大于枢轴，递归处理左右两边;
func quickSort(A []int, low, high int) {
	if low >= high {
		return
	}
	j := partition(A, low, high)
	quickSort(A, low, j)
	quickSort(A, j+1, high)
}


func partition(A []int, low, high int) int {
	pivot := A[low+(high-low)>>1]
	i, j := low-1, high+1
	for i < j {
		for i++; A[i] < pivot; i++ {
		}
		for j--; A[j] > pivot; j-- {
		}
		if i < j {
			A[i], A[j] = A[j], A[i]
		}
	}
	return j
}

func partition2(A []int, low, high int) int {
	pivot := A[low+(high-low)>>1]
	i, j := low-1, high+1
	for i < j {
		for {
			i++
			if A[i] >= pivot {
				break
			}
		}
		for {
			j--
			if A[j] <= pivot {
				break
			}
		}
		if i < j {
			A[i], A[j] = A[j], A[i]
		}
	}
	return j
}
```




## [手撕归并排序](https://leetcode.cn/problems/sort-an-array/)

```go
func sortArray(A []int) []int {
	MergeSort(A, 0, len(A)-1)
	return A
}

// 归并排序 O(nlogn)
// 是一种分而治之的算法，其思想是将一个列表分解为几个子列表，
// 直到每个子列表由一个元素组成，然后将这些子列表合并为排序后的列表。

// 先把数组从中间分成前后两部分，然后对前后两部分分别排序，
// 再将排好序的两部分合并在一起，这样整个数组就都有序了。
func MergeSort(A []int, start, end int) {
	if start >= end {
		return
	}
	mid := start + (end-start)>>1
	MergeSort(A, start, mid)
	MergeSort(A, mid+1, end)
	Arr := []int{}
	i, j := start, mid+1
	for i <= mid || j <= end {
		if j > end || (i <= mid && A[i] < A[j]) {
			Arr = append(Arr, A[i])
			i++
		} else {
			Arr = append(Arr, A[j])
			j++
		}
	}
	copy(A[start:end+1], Arr)
}

func MergeSort1(A []int, start, end int) {
	if start >= end {
		return
	}
	mid := start + (end-start)>>1 //分2部分定义当前数组
	MergeSort(A, start, mid)      //排序数组的第1部分
	MergeSort(A, mid+1, end)      //排序数组的第2部分
	Merge(A, start, mid, end)     //通过比较2个部分的元素来合并2个部分

}
func Merge(A []int, start, mid, end int) {
	Arr := make([]int, end-start+1)
	p, q, k := start, mid+1, 0
	for i := start; i <= end; i++ {
		if p > mid { //检查第一部分是否到达末尾
			Arr[k] = A[q]
			q++
		} else if q > end { //检查第二部分是否到达末尾
			Arr[k] = A[p]
			p++
		} else if A[p] <= A[q] { //检查哪一部分有更小的元素
			Arr[k] = A[p]
			p++
		} else {
			Arr[k] = A[q]
			q++
		}
		k++
	}
	for p := 0; p < k; p++ { // copy(A[start:end+1], Arr)
		A[start] = Arr[p]
		start++
	}
}

func Merge1(A []int, start, mid, end int) {
	Arr := make([]int, end-start+1)
	i, j, k := start, mid+1, 0
	for ; i <= mid && j <= end; k++ {
		if A[i] <= A[j] {
			Arr[k] = A[i]
			i++
		} else {
			Arr[k] = A[j]
			j++
		}
	}
	for ; i <= mid; i++ {
		Arr[k] = A[i]
		k++
	}
	for ; j <= end; j++ {
		Arr[k] = A[j]
		k++
	}
	copy(A[start:end+1], Arr)
}
```

## [手撕堆排序 ](https://leetcode-cn.com/problems/sort-an-array/)


**大根堆：升序**

```go
func sortArray(A []int) []int {
	heapSort(A)
	return A
}

// 在大根堆中、最大元素总在根上，堆排序使用堆的这个属性进行排序
func heapSort(A []int) {
	heapSize := len(A)
	buildMaxHeap(A, heapSize)
	for i := heapSize - 1; i >= 0; i-- {
		A[0], A[i] = A[i], A[0]    // 交换堆顶元素 A[0] 与堆底元素 A[i]，最大值 A[0] 放置在数组末尾
		heapSize--                 // 删除堆顶元素 A[0]
		maxHeapify(A, 0, heapSize) // 堆顶元素 A[0] 向下调整
	}
}

// 建堆 O(n)
func buildMaxHeap(A []int, heapSize int) {
	for i := heapSize >> 1; i >= 0; i-- { // heapSize / 2 后面都是叶子节点，不需要向下调整
		maxHeapify(A, i, heapSize)
	}
}

// 迭代: 调整大根堆 O(n)
func maxHeapify(A []int, i, heapSize int) {
	for i<<1+1 < heapSize { // i*2+1
		lson, rson, large := i<<1+1, i<<1+2, i
		if lson < heapSize && A[large] < A[lson] {
			large = lson
		}
		for rson < heapSize && A[large] < A[rson] {
			large = rson
		}
		if large != i {
			A[i], A[large] = A[large], A[i]
			i = large
		} else {
			break
		}
	}
}

// 递归: 调整大根堆 O(nlogn)
func MaxHeapify(A []int, i, heapSize int) {
	lson, rson, largest := i<<1+1, i<<1+2, i     // i*2+1, i*2+2
	if lson < heapSize && A[largest] < A[lson] { // 左儿子存在并大于根
		largest = lson
	}
	if rson < heapSize && A[largest] < A[rson] { // 右儿子存在并大于根
		largest = rson
	}
	if i != largest { // 找到左右儿子的最大值
		A[i], A[largest] = A[largest], A[i] // 堆顶调整为最大值
		MaxHeapify(A, largest, heapSize)    // 递归调整子树
	}
}
```


**小根堆：降序**
```go
func sortArray(A []int) []int {
	heapSort(A)
	return A
}

func heapSort(A []int) {
	heapSize := len(A)
	buildHeap(A, heapSize)
	for i := heapSize - 1; i >= 0; i-- {
		A[0], A[i] = A[i], A[0]
		heapSize--
		minHeapify(A, 0, heapSize)
	}
}

func buildHeap(A []int, heapSize int) {
	for i := heapSize >> 1; i >= 0; i-- {
		minHeapify(A, i, heapSize)
	}
}

// 小根堆：逆序
func minHeapify(A []int, i, heapSize int) {
	for i<<1+1 < heapSize {
		lson, rson, small := i<<1+1, i<<1+2, i
		for lson < heapSize && A[lson] < A[small] {
			small = lson
		}
		for rson < heapSize && A[rson] < A[small] {
			small = rson
		}
		if small != i {
			A[i], A[small] = A[small], A[i]
			i = small
		} else {
			break
		}
	}
}

func MinHeapify(A []int, i, heapSize int) {
	lson, rson, small := i<<1+1, i<<1+2, i
	for lson < heapSize && A[small] < A[lson] {
		small = lson
	}
	for rson < heapSize && A[small] < A[rson] {
		small = rson
	}
	if small != i {
		A[i], A[small] = A[small], A[i]
		MinHeapify(A, small, heapSize)
	}
}
```






```go
/*
 * @lc app=leetcode.cn id=912 lang=golang
 *
 * [912] 排序数组
 */

// @lc code=start
func sortArray(A []int) []int {
	InsertSort(A, len(A))
	SelectionSort(A, len(A))
	BubbleSort(A, len(A))
	
	return A
}

```

## 插入排序

```go
// 插入排序
// 取未排序区间中的元素，在已排序区间中找到合适的插入位置将其插入，并保证已排序区间数据一直有序
func InsertSort(A []int, n int) {
	for i := 0; i < n; i++ {
		temp, j := A[i], i // temp 插入元素
		for j > 0 && temp < A[j-1] { // 如果新元素小于有序元素
			A[j] = A[j-1] // 右移
			j--           // 向左扫描
		}
		A[j] = temp // 插入新元素
	}
}

func InsertSort1(A []int, n int) {
	for i := 1; i < n; i++ {
		value, j := A[i], i-1
		for j >= 0 && value < A[j] { // 查找插入位置
			A[j+1] = A[j] // 移动数据
			j--
		}
		A[j+1] = value
	}
}

func InsertSort2(A []int, n int) {
	for i := 0; i < n; i++ {
		insertElement := A[i] // 取无序的新元素
		insertPosition := i   // 插入位置
		for j := insertPosition - 1; j >= 0; j-- {
			if insertElement < A[j] { // 如果新元素小于有序的元素
				A[j+1] = A[j] // 有序的元素右移
				insertPosition--
			}
			A[insertPosition] = insertElement // 插入新元素
		}
	}
}
```

## 希尔排序

```go
// 希尔排序
// 先通过希尔增量逐步分组粗排，再插入排序
func ShellSort(A []int, n int) {
	for d := n >> 1; d > 0; d >>= 1 { // 希尔增量 d=n/2
		for i := d; i < n; i++ {
			temp, j := A[i], i
			for j >= d && temp < A[j-d] {
				A[j] = A[j-d] // 较大数右移一位
				j -= d        // 向左搜索
			}
			A[j] = temp // 插入
		}
	}
}

func ShellSort1(A []int, n int) {
	for d := n >> 1; d > 0; d >>= 1 { // 希尔增量 d=n/2
		for i := d; i < n; i++ {
			temp, j := A[i], i-d
			for j >= 0 && temp < A[j] {
				A[j+d] = A[j] // 较大数右移一位
				j -= d        // 向左搜索
			}
			A[j+d] = temp // 插入
		}
	}
}
```


## 选择排序

```go
// 选择排序(不稳定)
// 每次从未排序区间中找到最小的元素，将其放到已排序区间的末尾。
func SelectionSort(A []int, n int) {
	for i := 0; i < n; i++ {
		minIndex := i
		for j := i + 1; j < n; j++ {
			if A[j] < A[minIndex] {
				minIndex = j // 查找最小值下标
			}
		} // 将最小值交换到有序区
		A[i], A[minIndex] = A[minIndex], A[i]
	}
}
```

## 冒泡排序

```go
// 冒泡排序
// 从前往后遍历，如果前一个数大于后一个数，交换，一次冒泡一个元素已排序，重复n次。

func BubbleSort(A []int, n int) {
	for i := 0; i < n-1; i++ {
		for j := n - 1; j > i; j-- {
			if A[j-1] > A[j] {
				A[j-1], A[j] = A[j], A[j-1]
			}
		}
	}
}

func BubbleSort1(A []int, n int) {
	for i := 0; i < n-1; i++ {
		flag := false
		for j := n - 1; j > i; j-- {
			if A[j-1] > A[j] {
				A[j-1], A[j] = A[j], A[j-1]
				flag = true
			}
		}
		if !flag {
			break
		}
	}
}

func BubbleSort2(A []int, n int) {
	for i := 0; i < n-1; i++ {
		for j := 0; j < n-i-1; j++ {
			if A[j] > A[j+1] {
				A[j], A[j+1] = A[j+1], A[j]
			}
		}
	}
}

func BubbleSort3(A []int, n int) {
	if n <= 1 {
		return
	}
	for i := 0; i < n-1; i++ {
		flag := false // 提前退出冒泡循环的标志位
		for j := 0; j < n-i-1; j++ {
			if A[j] > A[j+1] {
				A[j], A[j+1] = A[j+1], A[j] // 交换相邻的2个元素
				flag = true                 // 表示有数据要交换
			}
		}
		if !flag { // 没有数据交换，提前退出
			break
		}
	}
}
```






## [Counting Sort](https://www.hackerearth.com/practice/algorithms/sorting/counting-sort/tutorial/)


```go

// 计数排序 模版
// 最大值是k，我们就可以把数据划分成k个桶。
// 每个桶内的数据值都是相同的，省掉了桶内排序的时间。
// [0, N-1]
func CountingSort(A []int) []int {
	// 首先找到数组的最大值 K
	K := 0
	for i := 0; i < len(A); i++ {
		if K < A[i] {
			K = A[i]
		}
	}
	// 存储数组 A 中每个元素的频率，值作为辅助数组的索引
	Aux := make([]int, K+1)
	for i := 0; i < len(A); i++ {
		Aux[A[i]]++
	}
	j := 0
	for i := 0; i <= K; i++ {
		tmp := Aux[i]
		for ; tmp > 0; tmp-- {
			A[j] = i
			j++
		}
	}
	return A
}

// 计数排序1 模版1
func CountingSort1(a []int, n int) {
	if n <= 1 {
		return
	}
	// 查找数组中数据的最大值
	var max int = math.MinInt32
	for i := range a {
		if a[i] > max {
			max = a[i]
		}
	}

	c := make([]int, max+1) // 申请一个计数数组c，下标大小[0,max]
	for i := range a {
		c[a[i]]++ // 计算每个元素的个数，放入c中
	}
	for i := 1; i <= max; i++ {
		c[i] += c[i-1] // 依次累加,c[i]存储小于等于i的个数
	}
	// 从数组C中取出下标为3的值7，也就是说，到目前为止，包括自己在内，分数小于等于3的考生有7个，
	// 也就是说3是数组R中的第7个元素（也就是数组R中下标为6的位置）。
	// 当3放入到数组R中后，小于等于3的元素就只剩下了6个了，所以相应的C[3]要减1，变成6。
	r := make([]int, n) // 临时数组r，存储排序之后的结果
	for i := n - 1; i >= 0; i-- {
		index := c[a[i]] - 1 //
		r[index] = a[i]
		c[a[i]]--
	}
	copy(a, r)
}

// 计数排序 实例
// -5 * 104 <= A[i] <= 5 * 104
func counting_sort(A []int) []int {
	K := 1000001          // 首先找到数组 A 的最大值 K
	Aux := make([]int, K) // 辅助数组 Aux 存储数组 A 中每个元素的频率，值作为辅助数组的索引
	for i := 0; i < len(A); i++ {
		Aux[A[i]+50000]++ // 防止-50000作为索引时，下标越界
	}
	for i, j := 0, 0; i < K; i++ {
		tmp := Aux[i] // 从小到大取数字i的频率
		for ; tmp > 0; tmp-- {
			A[j] = i - 50000
			j++
		}
	}
	return A
}
```






## [Bucket Sort](https://www.hackerearth.com/practice/algorithms/sorting/bucket-sort/tutorial/)

```go

// 桶排序

// 获取待排序数组中的最大值
func getMax(a []int) int {
	max := a[0]
	for i := 1; i < len(a); i++ {
		if a[i] > max {
			max = a[i]
		}
	}
	return max
}

func BucketSort(a []int) {
	num := len(a)
	if num <= 1 {
		return
	}
	max := getMax(a)
	buckets := make([][]int, num) // 二维切片

	index := 0
	for i := 0; i < num; i++ {
		index = a[i] * (num - 1) / max // 桶序号
		fmt.Println(index)
		buckets[index] = append(buckets[index], a[i]) // 加入对应的桶中
	}

	tmpPos := 0 // 标记数组位置
	for i := 0; i < num; i++ {
		bucketLen := len(buckets[i])
		if bucketLen > 0 {
			QuickSort(buckets[i], 0, len(buckets[i])-1) // 桶内做快速排序
			copy(a[tmpPos:], buckets[i])
			tmpPos += bucketLen
		}
	}

}

// 桶排序简单实现
func BucketSortSimple(source []int) {
	if len(source) <= 1 {
		return
	}
	array := make([]int, getMax(source)+1)
	for i := 0; i < len(source); i++ {
		array[source[i]]++
	}
	fmt.Println(array)
	c := make([]int, 0)
	for i := 0; i < len(array); i++ {
		for array[i] != 0 {
			c = append(c, i)
			array[i]--
		}
	}
	copy(source, c)

}


```


## [Radix Sort](https://www.hackerearth.com/practice/algorithms/sorting/radix-sort/tutorial/)


```go
// 基数排序
// 依次按个位、十位、百位进行排序
func RadixSort(A []int, n int) {

}
```









**10大排序总结**

---


```go
/*
 * @lc app=leetcode.cn id=912 lang=golang
 *
 * [912] 排序数组
 */

// @lc code=start
func sortArray(A []int) []int {
	// InsertSort(A, len(A))
	// InsertSort1(A, len(A)) // Time Limit Exceeded
	// InsertSort2(A, len(A)) // Time Limit Exceeded
	// ShellSort(A, len(A))
	// ShellSort1(A, len(A))
	// SelectionSort(A, len(A))
	// BubbleSort(A, len(A))

	rand.Seed(time.Now().UnixNano())
	// QuickSort(A, 0, len(A)-1)
	// QuickSort1(A, 0, len(A)-1)
	// MergeSort(A, 0, len(A)-1)
	// HeapSort(A)

	// CountingSort(A)             // index out of range [-1]
	// CountingSort1(A, len(A)) // index out of range [-1]
	// counting_sort(A)
	// bucketSort(A, len(A)-1)
	// BucketSortSimple(A)
	// RadixSort(A, len(A)-1) // 待做

	return A
}

// 快速排序
// 基于分而治之的方法，随机选择枢轴元素划分数组，
// 左边小于枢轴、右边大于枢轴，递归处理左右两边;
func QuickSort(A []int, start, end int) {
	if start >= end {
		return
	}
	x := A[(start+end)>>1] // x := A[(start+end)/2],用j划分递归子区间
	i, j := start-1, end+1 // 循环内直接扫描下一个数，导致多操作1次，所以预处理
	for i < j {
		for i++; A[i] < x; i++ { // 从左向右扫描，找到大于 x 的数，停止
		}
		for j--; A[j] > x; j-- { // 从右向左扫描，找到小于 x 的数，停止
		}
		if i < j {
			A[i], A[j] = A[j], A[i] // 交换, 使得左边小于 x, 右边大于 x
		}
	}
	QuickSort(A, start, j) // 递归处理 x 左边
	QuickSort(A, j+1, end) // 递归处理 x 右边
}

// 快速排序1
func QuickSort1(A []int, start, end int) {
	if start < end {
		piv_pos := randomPartition(A, start, end)
		QuickSort1(A, start, piv_pos-1)
		QuickSort1(A, piv_pos+1, end)
	}
}

func partition(A []int, start, end int) int {
	i, piv := start, A[end] // 从第一个数开始扫描，选取最后一位数字最为对比
	for j := start; j < end; j++ {
		if A[j] < piv { //  A[j]逆序：A[i] < piv < A[j]
			if i != j { // 不是同一个数
				A[i], A[j] = A[j], A[i] // A[j] 放在正确的位置
			}
			i++ //扫描下一个数
		}
	}
	A[i], A[end] = A[end], A[i] // A[end] 回到正确的位置
	return i
}

func randomPartition(A []int, start, end int) int {
	random := rand.Int()%(end-start+1) + start
	A[random], A[end] = A[end], A[random]
	return partition(A, start, end)
}

// 归并排序
// 是一种分而治之的算法，其思想是将一个列表分解为几个子列表，
// 直到每个子列表由一个元素组成，然后将这些子列表合并为排序后的列表。
func MergeSort(A []int, start, end int) {
	if start < end {
		mid := start + (end-start)>>1 //分2部分定义当前数组
		MergeSort(A, start, mid)      //排序数组的第1部分
		MergeSort(A, mid+1, end)      //排序数组的第2部分
		Merge(A, start, mid, end)     //通过比较2个部分的元素来合并2个部分
	}
}
func Merge(A []int, start, mid, end int) {
	Arr := make([]int, end-start+1)
	p, q, k := start, mid+1, 0
	for i := start; i <= end; i++ {
		if p > mid { //检查第一部分是否到达末尾
			Arr[k] = A[q]
			q++
		} else if q > end { //检查第二部分是否到达末尾
			Arr[k] = A[p]
			p++
		} else if A[p] <= A[q] { //检查哪一部分有更小的元素
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

func Merge1(A []int, start, mid, end int) {
	tmpArr := make([]int, end-start+1)
	i, j, k := start, mid+1, 0
	for ; i <= mid && j <= end; k++ {
		if A[i] <= A[j] {
			tmpArr[k] = A[i]
			i++
		} else {
			tmpArr[k] = A[j]
			j++
		}
	}
	for ; i <= mid; i++ {
		tmpArr[k] = A[i]
		k++
	}
	for ; j <= end; j++ {
		tmpArr[k] = A[j]
		k++
	}
	copy(A[start:end+1], tmpArr)
}

// 在大根堆中、最大元素总在根上，堆排序使用堆的这个属性进行排序
func HeapSort(A []int) {
	heap_size := len(A)
	build_maxheap(A, heap_size)
	for i := heap_size - 1; i >= 0; i-- {
		A[0], A[i] = A[i], A[0]      // 交换堆顶元素 A[0] 与堆底元素 A[i]，最大值 A[0] 放置在数组末尾
		heap_size--                  // 删除堆顶元素 A[0]
		max_heapify(A, 0, heap_size) // 堆顶元素 A[0] 向下调整
	}
}
func build_maxheap(A []int, heap_size int) { // 建堆 O(n)
	for i := heap_size >> 1; i >= 0; i-- { // heap_size / 2后面都是叶子节点，不需要向下调整
		max_heapify(A, i, heap_size)
	}
}
func max_heapify(A []int, i, heap_size int) { // 调整大根堆 O(nlogn)
	lson, rson, largest := i<<1+1, i<<1+2, i      // i*2+1, i*2+2
	if lson < heap_size && A[largest] < A[lson] { // 左儿子存在并大于根
		largest = lson
	}
	if rson < heap_size && A[largest] < A[rson] { // 右儿子存在并大于根
		largest = rson
	}
	if i != largest { // 找到左右儿子的最大值
		A[i], A[largest] = A[largest], A[i] // 堆顶调整为最大值
		max_heapify(A, largest, heap_size)  // 递归调整子树
	}
}

// 迭代
func maxHeapify(A []int, i, heapSize int) {
	for i<<1+1 < heapSize {
		lson, rson, large := i<<1+1, i<<1+2, i
		if lson < heapSize && A[large] < A[lson] {
			large = lson
		}
		for rson < heapSize && A[large] < A[rson] {
			large = rson
		}
		if large != i {
			A[i], A[large] = A[large], A[i]
			i = large
		} else {
			break
		}
	}
}

// 插入排序
// 取未排序区间中的元素，在已排序区间中找到合适的插入位置将其插入，并保证已排序区间数据一直有序
func InsertSort(A []int, n int) {
	for i := 0; i < n; i++ {
		temp, j := A[i], i
		for j > 0 && temp < A[j-1] { // 如果新元素小于有序元素
			A[j] = A[j-1] // 较大数右移一位
			j--           // 向左扫描
		}
		A[j] = temp // 插入新元素
	}
}

// Time Limit Exceeded
func InsertSort1(A []int, n int) {
	if n <= 1 {
		return
	}
	for i := 1; i < n; i++ {
		value, j := A[i], i-1
		for ; j >= 0 && value < A[j]; j-- { // 查找插入位置
			A[j+1] = A[j] // 移动数据
		}
		A[j+1] = value
	}
}

// Time Limit Exceeded
func InsertSort2(A []int, n int) {
	for i := 0; i < n; i++ {
		insertElement := A[i] // 取无序的新元素
		insertPosition := i   // 插入位置
		for j := insertPosition - 1; j >= 0; j-- {
			if insertElement < A[j] { // 如果新元素小于有序的元素
				A[j+1] = A[j] // 有序的元素右移
				insertPosition--
			}
			A[insertPosition] = insertElement // 插入新元素
		}
	}
}

// 希尔排序
// 先通过希尔增量逐步分组粗排，再插入排序
func ShellSort(A []int, n int) {
	for d := n >> 1; d > 0; d >>= 1 { // 希尔增量 d=n/2
		for i := d; i < n; i++ {
			temp, j := A[i], i
			for j >= d && temp < A[j-d] {
				A[j] = A[j-d] // 较大数右移一位
				j -= d        // 向左搜索
			}
			A[j] = temp // 插入
		}
	}
}

func ShellSort1(A []int, n int) {
	for d := n >> 1; d > 0; d >>= 1 { // 希尔增量 d=n/2
		for i := d; i < n; i++ {
			temp, j := A[i], i-d
			for j >= 0 && temp < A[j] {
				A[j+d] = A[j] // 较大数右移一位
				j -= d        // 向左搜索
			}
			A[j+d] = temp // 插入
		}
	}
}

// 选择排序(不稳定)
// 每次从未排序区间中找到最小的元素，将其放到已排序区间的末尾。
func SelectionSort(A []int, n int) {
	if n <= 1 {
		return
	}
	for i := 0; i < n; i++ {
		minIndex := i
		for j := i + 1; j < n; j++ {
			if A[j] < A[minIndex] {
				minIndex = j // 查找最小值下标
			}
		} // 将最小值交换到有序区
		A[i], A[minIndex] = A[minIndex], A[i]
	}
}

// 冒泡排序
// 从前往后遍历，如果前一个数大于后一个数，交换，一次冒泡一个元素已排序，重复n次。
func BubbleSort(A []int, n int) {
	if n <= 1 {
		return
	}
	for i := 0; i < n; i++ {
		flag := false // 提前退出冒泡循环的标志位
		for j := 0; j < n-i-1; j++ {
			if A[j] > A[j+1] {
				A[j], A[j+1] = A[j+1], A[j] // 交换相邻的2个元素
				flag = true                 // 表示有数据要交换
			}
		}
		if !flag { // 没有数据交换，提前退出
			break
		}
	}
}

// 桶排序
// 将要排序的数据分到几个有序的桶里，每个桶里的数据再单独进行排序。
// 桶内排完序之后，再把每个桶里的数据按照顺序依次取出，组成的序列就是有序的了。
func BucketSort(a []int) {
	num := len(a)
	if num <= 1 {
		return
	}
	max := getMax(a)
	buckets := make([][]int, num) // 二维切片

	index := 0
	for i := 0; i < num; i++ {
		index = a[i] * (num - 1) / max // 桶序号
		fmt.Println(index)
		buckets[index] = append(buckets[index], a[i]) // 加入对应的桶中
	}

	tmpPos := 0 // 标记数组位置
	for i := 0; i < num; i++ {
		bucketLen := len(buckets[i])
		if bucketLen > 0 {
			QuickSort(buckets[i], 0, len(buckets[i])-1) // 桶内做快速排序
			copy(a[tmpPos:], buckets[i])
			tmpPos += bucketLen
		}
	}

}

// 获取待排序数组中的最大值
func getMax(a []int) int {
	max := a[0]
	for i := 1; i < len(a); i++ {
		if a[i] > max {
			max = a[i]
		}
	}
	return max
}

// 桶排序简单实现
func BucketSortSimple(source []int) {
	if len(source) <= 1 {
		return
	}
	array := make([]int, getMax(source)+1)
	for i := 0; i < len(source); i++ {
		array[source[i]]++
	}
	fmt.Println(array)
	c := make([]int, 0)
	for i := 0; i < len(array); i++ {
		for array[i] != 0 {
			c = append(c, i)
			array[i]--
		}
	}
	copy(source, c)

}

// 计数排序 模版
// 最大值是k，我们就可以把数据划分成k个桶。
// 每个桶内的数据值都是相同的，省掉了桶内排序的时间。
// [0, N-1]
func CountingSort(A []int) []int {
	// 首先找到数组的最大值 K
	K := 0
	for i := 0; i < len(A); i++ {
		if K < A[i] {
			K = A[i]
		}
	}
	// 存储数组 A 中每个元素的频率，值作为辅助数组的索引
	Aux := make([]int, K+1)
	for i := 0; i < len(A); i++ {
		Aux[A[i]]++
	}
	j := 0
	for i := 0; i <= K; i++ {
		tmp := Aux[i]
		for ; tmp > 0; tmp-- {
			A[j] = i
			j++
		}
	}
	return A
}

// 计数排序1 模版1
func CountingSort1(a []int, n int) {
	if n <= 1 {
		return
	}
	// 查找数组中数据的最大值
	var max int = math.MinInt32
	for i := range a {
		if a[i] > max {
			max = a[i]
		}
	}

	c := make([]int, max+1) // 申请一个计数数组c，下标大小[0,max]
	for i := range a {
		c[a[i]]++ // 计算每个元素的个数，放入c中
	}
	for i := 1; i <= max; i++ {
		c[i] += c[i-1] // 依次累加,c[i]存储小于等于i的个数
	}
	// 从数组C中取出下标为3的值7，也就是说，到目前为止，包括自己在内，分数小于等于3的考生有7个，
	// 也就是说3是数组R中的第7个元素（也就是数组R中下标为6的位置）。
	// 当3放入到数组R中后，小于等于3的元素就只剩下了6个了，所以相应的C[3]要减1，变成6。
	r := make([]int, n) // 临时数组r，存储排序之后的结果
	for i := n - 1; i >= 0; i-- {
		index := c[a[i]] - 1 //
		r[index] = a[i]
		c[a[i]]--
	}
	copy(a, r)
}

// 计数排序 实例
// -5 * 104 <= A[i] <= 5 * 104
func counting_sort(A []int) []int {
	K := 1000001          // 首先找到数组 A 的最大值 K
	Aux := make([]int, K) // 辅助数组 Aux 存储数组 A 中每个元素的频率，值作为辅助数组的索引
	for i := 0; i < len(A); i++ {
		Aux[A[i]+50000]++ // 防止-50000作为索引时，下标越界
	}
	for i, j := 0, 0; i < K; i++ {
		tmp := Aux[i] // 从小到大取数字i的频率
		for ; tmp > 0; tmp-- {
			A[j] = i - 50000
			j++
		}
	}
	return A
}

// 基数排序
// 依次按个位、十位、百位进行排序
func RadixSort(A []int, n int) {

}

// @lc code=end


```








**参考记录**
---

## [补充题4. 手撕快速排序 ](https://leetcode-cn.com/problems/sort-an-array/)


```go

/*
 * @lc app=leetcode.cn id=912 lang=golang
 *
 * [912] 排序数组
 */

// @lc code=start
func sortArray(A []int) []int {
    QuickSort(A, 0, len(A)-1)

	rand.Seed(time.Now().UnixNano())
    //QuickSort1(A, 0, len(A)-1)
    
    return A
}
// 快速排序 O(nlogn)
// 基于分而治之的方法，随机选择枢轴元素划分数组，
// 左边小于枢轴、右边大于枢轴，递归处理左右两边;
func QuickSort(A []int, start, end int) {
	if start >= end {
		return
	}
	x := A[(start+end)>>1] // x := A[(start+end)/2],用j划分递归子区间
	i, j := start-1, end+1 // 循环内直接扫描下一个数，导致多操作1次，所以预处理
	for i < j {
		for i++; A[i] < x; i++ { // 从左向右扫描，找到大于 x 的数，停止
		}
		for j--; A[j] > x; j-- { // 从右向左扫描，找到小于 x 的数，停止
		}
		if i < j {
			A[i], A[j] = A[j], A[i] // 交换, 使得左边小于 x, 右边大于 x
		}
	}
	QuickSort(A, start, j) // 递归处理 x 左边
	QuickSort(A, j+1, end) // 递归处理 x 右边
}

// 快速排序1
func QuickSort1(A []int, start, end int) {
	if start < end {
		piv_pos := randomPartition(A, start, end)
		QuickSort1(A, start, piv_pos-1)
		QuickSort1(A, piv_pos+1, end)
	}
}

func partition(A []int, start, end int) int {
	i, piv := start, A[end] // 从第一个数开始扫描，选取最后一位数字最为对比
	for j := start; j < end; j++ {
		if A[j] < piv { //  A[j]逆序：A[i] < piv < A[j]
			if i != j { // 不是同一个数
				A[i], A[j] = A[j], A[i] // A[j] 放在正确的位置
			}
			i++ //扫描下一个数
		}
	}
	A[i], A[end] = A[end], A[i] // A[end] 回到正确的位置
	return i
}

func randomPartition(A []int, start, end int) int {
	random := rand.Int()%(end-start+1) + start
	A[random], A[end] = A[end], A[random]
	return partition(A, start, end)
}
```


## [补充题5. 手撕归并排序](https://leetcode.cn/problems/sort-an-array/)

```go
func sortArray(A []int) []int {
	MergeSort(A, 0, len(A)-1)
	return A
}

// 归并排序 O(nlogn)
// 是一种分而治之的算法，其思想是将一个列表分解为几个子列表，
// 直到每个子列表由一个元素组成，然后将这些子列表合并为排序后的列表。

// 先把数组从中间分成前后两部分，然后对前后两部分分别排序，
// 再将排好序的两部分合并在一起，这样整个数组就都有序了。
func MergeSort(A []int, start, end int) {
	if start < end {
		mid := start + (end-start)>>1 //分2部分定义当前数组
		MergeSort(A, start, mid)      //排序数组的第1部分
		MergeSort(A, mid+1, end)      //排序数组的第2部分
		Merge(A, start, mid, end)     //通过比较2个部分的元素来合并2个部分
	}
}
func Merge(A []int, start, mid, end int) {
	Arr := make([]int, end-start+1)
	p, q, k := start, mid+1, 0
	for i := start; i <= end; i++ {
		if p > mid { //检查第一部分是否到达末尾
			Arr[k] = A[q]
			q++
		} else if q > end { //检查第二部分是否到达末尾
			Arr[k] = A[p]
			p++
		} else if A[p] <= A[q] { //检查哪一部分有更小的元素
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

func Merge1(A []int, start, mid, end int) {
	tmpArr := make([]int, end-start+1)
	i, j, k := start, mid+1, 0
	for ; i <= mid && j <= end; k++ {
		if A[i] <= A[j] {
			tmpArr[k] = A[i]
			i++
		} else {
			tmpArr[k] = A[j]
			j++
		}
	}
	for ; i <= mid; i++ {
		tmpArr[k] = A[i]
		k++
	}
	for ; j <= end; j++ {
		tmpArr[k] = A[j]
		k++
	}
	copy(A[start:end+1], tmpArr)
}
```

## [补充题6. 手撕堆排序 ](https://leetcode-cn.com/problems/sort-an-array/)

**大根堆：顺序**

```go
func sortArray(A []int) []int {
	heapSort(A)
	return A
}

// 在大根堆中、最大元素总在根上，堆排序使用堆的这个属性进行排序
func heapSort(A []int) {
	heapSize := len(A)
	buildMaxHeap(A, heapSize)
	for i := heapSize - 1; i >= 0; i-- {
		A[0], A[i] = A[i], A[0]    // 交换堆顶元素 A[0] 与堆底元素 A[i]，最大值 A[0] 放置在数组末尾
		heapSize--                 // 删除堆顶元素 A[0]
		maxHeapify(A, 0, heapSize) // 堆顶元素 A[0] 向下调整
	}
}

// 建堆 O(n)
func buildMaxHeap(A []int, heapSize int) {
	for i := heapSize >> 1; i >= 0; i-- { // heapSize / 2 后面都是叶子节点，不需要向下调整
		maxHeapify(A, i, heapSize)
	}
}

// 迭代: 调整大根堆 O(n)
func maxHeapify(A []int, i, heapSize int) {
	for i<<1+1 < heapSize { // i*2+1
		lson, rson, large := i<<1+1, i<<1+2, i
		if lson < heapSize && A[large] < A[lson] {
			large = lson
		}
		for rson < heapSize && A[large] < A[rson] {
			large = rson
		}
		if large != i {
			A[i], A[large] = A[large], A[i]
			i = large
		} else {
			break
		}
	}
}

// 递归: 调整大根堆 O(nlogn)
func MaxHeapify(A []int, i, heapSize int) {
	lson, rson, largest := i<<1+1, i<<1+2, i     // i*2+1, i*2+2
	if lson < heapSize && A[largest] < A[lson] { // 左儿子存在并大于根
		largest = lson
	}
	if rson < heapSize && A[largest] < A[rson] { // 右儿子存在并大于根
		largest = rson
	}
	if i != largest { // 找到左右儿子的最大值
		A[i], A[largest] = A[largest], A[i] // 堆顶调整为最大值
		MaxHeapify(A, largest, heapSize)    // 递归调整子树
	}
}
```


**小根堆：逆序**
```go
func sortArray(A []int) []int {
	heapSort(A)
	return A
}

func heapSort(A []int) {
	heapSize := len(A)
	buildHeap(A, heapSize)
	for i := heapSize - 1; i >= 0; i-- {
		A[0], A[i] = A[i], A[0]
		heapSize--
		minHeapify(A, 0, heapSize)
	}
}

func buildHeap(A []int, heapSize int) {
	for i := heapSize >> 1; i >= 0; i-- {
		minHeapify(A, i, heapSize)
	}
}

// 小根堆：逆序
func minHeapify(A []int, i, heapSize int) {
	for i<<1+1 < heapSize {
		lson, rson, small := i<<1+1, i<<1+2, i
		for lson < heapSize && A[lson] < A[small] {
			small = lson
		}
		for rson < heapSize && A[rson] < A[small] {
			small = rson
		}
		if small != i {
			A[i], A[small] = A[small], A[i]
			i = small
		} else {
			break
		}
	}
}

func MinHeapify(A []int, i, heapSize int) {
	lson, rson, small := i<<1+1, i<<1+2, i
	for lson < heapSize && A[small] < A[lson] {
		small = lson
	}
	for rson < heapSize && A[small] < A[rson] {
		small = rson
	}
	if small != i {
		A[i], A[small] = A[small], A[i]
		MinHeapify(A, small, heapSize)
	}
}
```


## 插入、选择、冒泡排序

```go
/*
 * @lc app=leetcode.cn id=912 lang=golang
 *
 * [912] 排序数组
 */

// @lc code=start
func sortArray(A []int) []int {
	InsertSort(A, len(A))
	SelectionSort(A, len(A))
	BubbleSort(A, len(A))
	
	return A
}


// 插入排序
// 取未排序区间中的元素，在已排序区间中找到合适的插入位置将其插入，并保证已排序区间数据一直有序
func InsertSort(A []int, n int) {
	for i := 0; i < n; i++ {
		temp, j := A[i], i
		for j > 0 && temp < A[j-1] { // 如果新元素小于有序元素
			A[j] = A[j-1] // 右移
			j--           // 向左扫描
		}
		A[j] = temp // 插入新元素
	}
}

// 选择排序(不稳定)
// 每次从未排序区间中找到最小的元素，将其放到已排序区间的末尾。
func SelectionSort(A []int, n int) {
	if n <= 1 {
		return
	}
	for i := 0; i < n; i++ {
		minIndex := i
		for j := i + 1; j < n; j++ {
			if A[j] < A[minIndex] {
				minIndex = j // 查找最小值下标
			}
		} // 将最小值交换到有序区
		A[i], A[minIndex] = A[minIndex], A[i]
	}
}

// 冒泡排序
// 从前往后遍历，如果前一个数大于后一个数，交换，一次冒泡一个元素已排序，重复n次。
func BubbleSort(A []int, n int) {
	if n <= 1 {
		return
	}
	for i := 0; i < n; i++ {
		flag := false // 提前退出冒泡循环的标志位
		for j := 0; j < n-i-1; j++ {
			if A[j] > A[j+1] {
				A[j], A[j+1] = A[j+1], A[j] // 交换相邻的2个元素
				flag = true                 // 表示有数据要交换
			}
		}
		if !flag { // 没有数据交换，提前退出
			break
		}
	}
}
```


```go
func InsertSort(A []int, n int) {
	if n <= 1 {
		return
	}
	for i := 1; i < n; i++ {
		value, j := A[i], i-1
		for ; j >= 0 && value < A[j]; j-- { // 查找插入位置
			A[j+1] = A[j] // 移动数据
		}
		A[j+1] = value
	}
}

func InsertSort1(A []int, n int) {
	for i := 0; i < n; i++ {
		insertElement := A[i] // 取无序的新元素
		insertPosition := i   // 插入位置
		for j := insertPosition - 1; j >= 0; j-- {
			if insertElement < A[j] { // 如果新元素小于有序的元素
				A[j+1] = A[j] // 有序的元素右移
				insertPosition--
			}
			A[insertPosition] = insertElement // 插入新元素
		}
	}
}
```




## [Counting Sort](https://www.hackerearth.com/practice/algorithms/sorting/counting-sort/tutorial/)


```go

// 计数排序 模版
// 最大值是k，我们就可以把数据划分成k个桶。
// 每个桶内的数据值都是相同的，省掉了桶内排序的时间。
// [0, N-1]
func CountingSort(A []int) []int {
	// 首先找到数组的最大值 K
	K := 0
	for i := 0; i < len(A); i++ {
		if K < A[i] {
			K = A[i]
		}
	}
	// 存储数组 A 中每个元素的频率，值作为辅助数组的索引
	Aux := make([]int, K+1)
	for i := 0; i < len(A); i++ {
		Aux[A[i]]++
	}
	j := 0
	for i := 0; i <= K; i++ {
		tmp := Aux[i]
		for ; tmp > 0; tmp-- {
			A[j] = i
			j++
		}
	}
	return A
}

// 计数排序1 模版1
func CountingSort1(a []int, n int) {
	if n <= 1 {
		return
	}
	// 查找数组中数据的最大值
	var max int = math.MinInt32
	for i := range a {
		if a[i] > max {
			max = a[i]
		}
	}

	c := make([]int, max+1) // 申请一个计数数组c，下标大小[0,max]
	for i := range a {
		c[a[i]]++ // 计算每个元素的个数，放入c中
	}
	for i := 1; i <= max; i++ {
		c[i] += c[i-1] // 依次累加,c[i]存储小于等于i的个数
	}
	// 从数组C中取出下标为3的值7，也就是说，到目前为止，包括自己在内，分数小于等于3的考生有7个，
	// 也就是说3是数组R中的第7个元素（也就是数组R中下标为6的位置）。
	// 当3放入到数组R中后，小于等于3的元素就只剩下了6个了，所以相应的C[3]要减1，变成6。
	r := make([]int, n) // 临时数组r，存储排序之后的结果
	for i := n - 1; i >= 0; i-- {
		index := c[a[i]] - 1 //
		r[index] = a[i]
		c[a[i]]--
	}
	copy(a, r)
}

// 计数排序 实例
// -5 * 104 <= A[i] <= 5 * 104
func counting_sort(A []int) []int {
	K := 1000001          // 首先找到数组 A 的最大值 K
	Aux := make([]int, K) // 辅助数组 Aux 存储数组 A 中每个元素的频率，值作为辅助数组的索引
	for i := 0; i < len(A); i++ {
		Aux[A[i]+50000]++ // 防止-50000作为索引时，下标越界
	}
	for i, j := 0, 0; i < K; i++ {
		tmp := Aux[i] // 从小到大取数字i的频率
		for ; tmp > 0; tmp-- {
			A[j] = i - 50000
			j++
		}
	}
	return A
}
```






## [Bucket Sort](https://www.hackerearth.com/practice/algorithms/sorting/bucket-sort/tutorial/)

```go

// 桶排序

// 获取待排序数组中的最大值
func getMax(a []int) int {
	max := a[0]
	for i := 1; i < len(a); i++ {
		if a[i] > max {
			max = a[i]
		}
	}
	return max
}

func BucketSort(a []int) {
	num := len(a)
	if num <= 1 {
		return
	}
	max := getMax(a)
	buckets := make([][]int, num) // 二维切片

	index := 0
	for i := 0; i < num; i++ {
		index = a[i] * (num - 1) / max // 桶序号
		fmt.Println(index)
		buckets[index] = append(buckets[index], a[i]) // 加入对应的桶中
	}

	tmpPos := 0 // 标记数组位置
	for i := 0; i < num; i++ {
		bucketLen := len(buckets[i])
		if bucketLen > 0 {
			QuickSort(buckets[i], 0, len(buckets[i])-1) // 桶内做快速排序
			copy(a[tmpPos:], buckets[i])
			tmpPos += bucketLen
		}
	}

}

// 桶排序简单实现
func BucketSortSimple(source []int) {
	if len(source) <= 1 {
		return
	}
	array := make([]int, getMax(source)+1)
	for i := 0; i < len(source); i++ {
		array[source[i]]++
	}
	fmt.Println(array)
	c := make([]int, 0)
	for i := 0; i < len(array); i++ {
		for array[i] != 0 {
			c = append(c, i)
			array[i]--
		}
	}
	copy(source, c)

}

func QuickSort(A []int, start, end int) {
	if start >= end {
		return
	}
	pivot := A[start+(end-start)>>1]
	i, j := start-1, end+1
	for i < j {
		for i++; A[i] < pivot; i++ {
		}
		for j--; A[j] > pivot; j-- {
		}
		if i < j {
			A[i], A[j] = A[j], A[i]
		}
	}
	QuickSort(A, start, j)
	QuickSort(A, j+1, end)
}
```


## [Radix Sort](https://www.hackerearth.com/practice/algorithms/sorting/radix-sort/tutorial/)


```go
// 基数排序
// 依次按个位、十位、百位进行排序
func RadixSort(A []int, n int) {

}
```





# 2022 Sorting

## 1. [Quick Sort](https://www.hackerearth.com/practice/algorithms/sorting/quick-sort/tutorial/)


快速排序基于分而治之的方法，随机选择枢轴元素划分数组，左边小于枢轴、右边大于枢轴，递归处理左右两边


```go
func sortArray(A []int) []int {
	quickSort(A, 0, len(A)-1)
	return A
}

func quickSort(A []int, start, end int) {
	if start >= end {
		return
	}
	x := A[(start+end)>>1]  // x := A[(start+end)/2],用j划分递归子区间
	i, j := start-1, end+1 // 循环内直接扫描下一个数，导致多操作1次，所以预处理
	for i < j {
		for i++; A[i] < x; i++ { // 从左向右扫描，找到大于 x 的数，停止
		}
		for j--; A[j] > x; j-- { // 从右向左扫描，找到小于 x 的数，停止
		}
		if i < j {
			A[i], A[j] = A[j], A[i] // 交换, 使得左边小于 x, 右边大于 x
		}
	}
	quickSort(A, start, j) // 递归处理 x 左边
	quickSort(A, j+1, end) // 递归处理 x 右边
}
```

**解法二**

选取最后一个数字作为中枢
```go
func sortArray(A []int) []int {
    rand.Seed(time.Now().UnixNano())
    quick_sort(A, 0, len(A)-1)
    return A
}

func quick_sort(A []int, start, end int) {
    if start < end {
        piv_pos := random_partition(A, start, end)
        quick_sort(A, start, piv_pos-1)
        quick_sort(A, piv_pos+1, end)
    }
}

// 选取最后一个数字作为中枢
func partition(A []int, start, end int) int {
    i, piv := start, A[end] // 从第一个数开始扫描，选取最后一位数字最为对比
    for j := start; j < end; j++ {
        if A[j] < piv { // A[i] < piv < A[j]
            if i != j { // 不是同一个数
                A[i], A[j] = A[j], A[i]// A[j] 放在正确的位置
            }
            i++//扫描下一个数
        }
    }
    A[i], A[end] = A[end], A[i] // A[end] 回到正确的位置
    return i 
}

func random_partition(A []int, start, end int) int {
    random := rand.Int()%(end-start+1)+start
    A[random], A[end] = A[end],A[random]
    return partition(A, start, end)
}
```

选取第一个数字作为中枢
```go
func quick_sort(A []int, start, end int) {
	if start < end {
		piv_pos := random_partition(A, start, end)
		quick_sort(A, start, piv_pos-1)
		quick_sort(A, piv_pos+1, end)
	}
}

// 选取第一个数字作为中枢
func partition(A []int, start, end int) int {
	piv, i := A[start], start+1//第一个元素作为枢轴
	for j := start + 1; j <= end; j++ {
		if A[j] < piv {//小于枢轴的放一边、大于枢轴的放另一边
			A[i], A[j] = A[j], A[i]
			i++
		}
	}
	A[start], A[i-1] = A[i-1], A[start] //放置枢轴到正确的位置
	return i - 1 						//返回枢轴的位置
}
func random_partition(A []int, start, end int) int {
	rand.Seed(time.Now().UnixNano())
	random := start + rand.Int()%(end-start+1)
	A[start], A[random] = A[random], A[start]
	return partition(A, start, end)
}
```




## 2. [Heap Sort](https://www.hackerearth.com/practice/algorithms/sorting/heap-sort/tutorial/)

在大根堆中、最大元素总在根上，堆排序使用堆的这个属性进行排序

```go
func heap_sort(A []int) {
	heap_size := len(A)
	build_maxheap(A, heap_size)
	for i := heap_size - 1; i >= 0; i-- {
		A[0], A[i] = A[i], A[0]      // 交换堆顶元素 A[0] 与堆底元素 A[i]，最大值 A[0] 放置在数组末尾
		heap_size--                  // 删除堆顶元素 A[0]
		max_heapify(A, 0, heap_size) // 堆顶元素 A[0] 向下调整
	}
}
func build_maxheap(A []int, heap_size int) { // 建堆 O(n)
	for i := heap_size / 2; i >= 0; i-- {   // heap_size>>1 后面都是叶子节点，不需要向下调整
		max_heapify(A, i, heap_size)
	}
}
// 递归
func max_heapify(A []int, i, heap_size int) {     // 调整大根堆 O(nlogn)
	lson, rson, largest := i*2+1, i*2+2, i	  // i<<1+1, i<<1+2
	if lson < heap_size && A[largest] < A[lson] { // 左儿子存在并大于根
		largest = lson
	}
	if rson < heap_size && A[largest] < A[rson] { // 右儿子存在并大于根
		largest = rson
	}
	if i != largest {                       // 找到左右儿子的最大值
		A[i], A[largest] = A[largest], A[i] // 堆顶调整为最大值
		max_heapify(A, largest, heap_size)  // 递归调整子树
	}
}

// 迭代
func maxHeapify(A []int, i, heapSize int) {
	for i<<1+1 < heapSize {
		lson, rson, large := i<<1+1, i<<1+2, i
		if lson < heapSize && A[large] < A[lson] {
			large = lson
		}
		for rson < heapSize && A[large] < A[rson] {
			large = rson
		}
		if large != i {
			A[i], A[large] = A[large], A[i]
			i = large
		} else {
			break
		}
	}
}
```

## 3. [Merge Sort](https://www.hackerearth.com/practice/algorithms/sorting/merge-sort/tutorial/)

归并排序是一种分而治之的算法，其思想是将一个列表分解为几个子列表，直到每个子列表由一个元素组成，然后将这些子列表合并为排序后的列表。

```go
func merge_sort(A []int, start, end int) {
	if start < end {
		mid := start + (end-start)>>1 // 分2部分定义当前数组
		merge_sort(A, start, mid)     // 排序数组的第1部分
		merge_sort(A, mid+1, end)     // 排序数组的第2部分
		merge(A, start, mid, end)     // 通过比较2个部分的元素来合并2个部分
	}
}
func merge(A []int, start, mid, end int) {
	Arr := make([]int, end-start+1)
	p, q, k := start, mid+1, 0
	for i := start; i <= end; i++ {
		if p > mid { // 检查第一部分是否到达末尾
			Arr[k] = A[q]
			q++
		} else if q > end { // 检查第二部分是否到达末尾
			Arr[k] = A[p]
			p++
		} else if A[p] <= A[q] { // 检查哪一部分有更小的元素
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


func Merge1(A []int, start, mid, end int) {
	tmpArr := make([]int, end-start+1)
	i, j, k := start, mid+1, 0
	for ; i <= mid && j <= end; k++ {
		if A[i] <= A[j] {
			tmpArr[k] = A[i]
			i++
		} else {
			tmpArr[k] = A[j]
			j++
		}
	}
	for ; i <= mid; i++ {
		tmpArr[k] = A[i]
		k++
	}
	for ; j <= end; j++ {
		tmpArr[k] = A[j]
		k++
	}
	copy(A[start:end+1], tmpArr)
}
```

## 4. [Insertion Sort](https://www.hackerearth.com/practice/algorithms/sorting/insertion-sort/tutorial/#c252800)

插入排序基于这样的想法：每次迭代都会消耗输入元素中的一个元素，以找到其正确位置，即该元素在排序数组中的位置。

通过在每次迭代时增加排序后的数组来迭代输入元素。它将当前元素与已排序数组中的最大值进行比较。
如果当前元素更大，则它将元素留在其位置，然后移至下一个元素，否则它将在已排序数组中找到其正确位置，并将其移至该位置。
这是通过将已排序数组中所有大于当前元素的元素移动到前面的一个位置来完成的

```go
func insertion_sort(A []int, n int) {
	for i := 0; i < n; i++ {
		temp, j := A[i], i
		for j > 0 && temp < A[j-1] { //当前元素小于左边元素
			A[j] = A[j-1] //向前移动左边元素
			j--
		}
		A[j] = temp //移动当前元素到正确的位置 A[j-1] < temp A[j] 
	}
}
```






## 5. [Bubble Sort](https://www.hackerearth.com/practice/algorithms/sorting/bubble-sort/tutorial/)


反复比较成对的相邻元素，交换它们的位置如果他们在无序区。（最大元素冒泡到最后）

```go
func bubble_sort(A []int, n int) {
	for k := 0; k < n-1; k++ {  // (n-k-1) 是忽略比较的元素，这些元素已比较完成在简单的迭代中
		for i := 0; i < n-k-1; i++ {
			if A[i] > A[i+1] {
				A[i], A[i+1] = A[i+1], A[i] //交换
			}
		}
	}
}
```


## 6. [Selection Sort](https://www.hackerearth.com/practice/algorithms/sorting/selection-sort/tutorial/)

在未排序的数组中找到最小或最大元素，然后将其放在已排序的数组中的正确位置。

```go
func selection_sort(A []int, n int) {
	for i := 0; i < n-1; i++ {		 //在每次迭代中将数组的有效大小减少1
		min := i                     //假设第一个元素是未排序数组的最小值
		for j := i + 1; j < n; j++ { //给出未排序数组的有效大小
			if A[j] < A[min] { //找到最小的元素
				min = j
			}
		}
		A[i], A[min] = A[min], A[i] //将最小元素放在适当的位置
	}
}
```

## 7.[Counting Sort](https://www.hackerearth.com/practice/algorithms/sorting/counting-sort/tutorial/)



```go

// 计数排序1 模版
// [0, N-1]
func counting_sort1(A []int) []int {
	// 首先找到数组的最大值 K
	K := 0
	for i := 0; i < len(A); i++ {
		if K < A[i] {
			K = A[i]
		}
	}
	// 存储数组 A 中每个元素的频率，值作为辅助数组的索引
	Aux := make([]int, K+1)
	for i := 0; i < len(A); i++ {
		Aux[A[i]]++
	}
	j := 0
	for i := 0; i <= K; i++ {
		tmp := Aux[i]
		for ; tmp > 0; tmp-- {
			A[j] = i
			j++
		}
	}
	return A
}

// 计数排序
// -5 * 104 <= A[i] <= 5 * 104
func counting_sort(A []int) []int {
	K := 1000001          // 首先找到数组 A 的最大值 K
	Aux := make([]int, K) // 辅助数组 Aux 存储数组 A 中每个元素的频率，值作为辅助数组的索引
	for i := 0; i < len(A); i++ {
		Aux[A[i]+50000]++ // 防止-50000作为索引时，下标越界
	}
	for i, j := 0, 0; i < K; i++ {
		tmp := Aux[i] // 从小到大取数字i的频率
		for ; tmp > 0; tmp-- {
			A[j] = i - 50000
			j++
		}
	}
	return A
}
```



## [Radix Sort](https://www.hackerearth.com/practice/algorithms/sorting/radix-sort/tutorial/)




## [Bucket Sort](https://www.hackerearth.com/practice/algorithms/sorting/bucket-sort/tutorial/)














---


## Sorting

### 1. Quick Sort



```go
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
	A[i-1], A[start] = A[start], A[i-1]
	return i - 1
}
func random_partition(A []int, start, end int) int {
	rand.Seed(time.Now().Unix())
	random := start + rand.Int()%(end-start+1)
	A[start], A[random] = A[random], A[start]
	return partition(A, start, end)
}
```

```go
func quick_sort(A []int, start, end int) {
	if start < end {
		piv_pos := random_partition(A, start, end)
		quick_sort(A, start, piv_pos-1)
		quick_sort(A, piv_pos+1, end)
	}
}
func partition(A []int, start, end int) int {
	piv, i := A[end], start-1
	for j := start; j < end; j++ {
		if A[j] < piv {
			i++
			A[i], A[j] = A[j], A[i]
		}
	}
	A[i+1], A[end] = A[end], A[i+1]
	return i + 1
}
func random_partition(A []int, start, end int) int {
	random := start + rand.Int()%(end-start+1)
	A[random], A[end] = A[end], A[random]
	return partition(A, start, end)
}
```

### 2. Heap Sort

```go
func heap_sort(A []int) {
	heap_size := len(A)
	build_maxheap(A, heap_size)
	for i := heap_size - 1; i >= 0; i-- {
		A[0], A[i] = A[i], A[0]
		heap_size--
		heapify(A, 0, heap_size)
	}
}
func build_maxheap(A []int, heap_size int) {
	for i := heap_size >> 1; i >= 0; i-- {
		heapify(A, i, heap_size)
	}
}
func heapify(A []int, i, heap_size int) {
	lson, rson, largest := i<<1+1, i<<1+2, i
	if lson < heap_size && A[largest] < A[lson] {
		largest = lson
	}
	if rson < heap_size && A[largest] < A[rson] {
		largest = rson
	}
	if i != largest {
		A[i], A[largest] = A[largest], A[i]
		heapify(A, largest, heap_size)
	}
}
```


### 3. Merge Sort

```go
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

### 4. Insertion Sort


```go
func insertion_sort(A []int, n int) {
	for i := 0; i < n; i++ {
		temp, j := A[i], i
		for j > 0 && temp < A[j-1] {
			A[j] = A[j-1]
			j--
		}
		A[j] = temp
	}
}
```

### 5. Bubble Sort

```go
func bubble_sort(A []int, n int) {
	for k := 0; k < n; k++ {
		for i := 0; i < n-k-1; i++ {
			if A[i] > A[i+1] {
				A[i], A[i+1] = A[i+1], A[i]
			}
		}
	}
}
```


### 6. Selection Sort


```go
func selection_sort(A []int, n int) {
	for i := 0; i < n-1; i++ {
		min := i
		for j := i + 1; j < n; j++ {
			if A[j] < A[min] {
				min = j
			}
		}
		A[i], A[min] = A[min], A[i]
	}
}
```




![常见的排序算法的时间复杂度.png](http://ww1.sinaimg.cn/large/007daNw2ly1go45x0kga6j32a017e4cd.jpg)


------

### 排序高频题

[补充题4. 手撕快速排序 912. 排序数组](https://leetcode-cn.com/problems/sort-an-array/)

[215. 数组中的第K个最大元素](https://leetcode-cn.com/problems/kth-largest-element-in-an-array/)

[补充题6. 手撕堆排序 912. 排序数组](https://leetcode-cn.com/problems/sort-an-array/)

[补充题5. 手撕归并排序 912. 排序数组](https://leetcode-cn.com/problems/sort-an-array/)

[785. 快速排序](https://www.acwing.com/problem/content/description/787/)

[Quick Sort](https://www.hackerearth.com/practice/algorithms/sorting/quick-sort/tutorial/)
------







[补充题4. 手撕快速排序 912. 排序数组](https://leetcode-cn.com/problems/sort-an-array/)



```go
func sortArray(A []int) []int {
	quick_sort(A, 0, len(A)-1)
	return A
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


复杂度分析

- 时间复杂度：基于随机选取主元的快速排序时间复杂度为期望 O(nlogn)，其中 n 为数组的长度。详细证明过程可以见《算法导论》第七章，这里不再大篇幅赘述。

- 空间复杂度：O(h)，其中 h 为快速排序递归调用的层数。我们需要额外的 O(h) 的递归调用的栈空间，由于划分的结果不同导致了快速排序递归调用的层数也会不同，最坏情况下需 O(n) 的空间，最优情况下每次都平衡，此时整个递归树高度为 logn，空间复杂度为 O(logn)。




[215. 数组中的第K个最大元素](https://leetcode-cn.com/problems/kth-largest-element-in-an-array/)

## 方法一：基于快速排序的选择方法

快速选择算法思路：

只要某次划分的 q 为倒数第 k 个下标的时候，我们就已经找到了答案。
如果划分得到的 q 正好就是我们需要的下标，就直接返回 a[q]；
否则，如果 q 比目标下标小，就递归右子区间，否则递归左子区间。

```go
func findKthLargest(A []int, k int) int {
	rand.Seed(time.Now().Unix())
	n := len(A)
	return quick_select(A, 0, n-1, n-k)
}
func quick_select(A []int, start, end, i int) int {
	piv_pos := random_partition(A, start, end)
	if piv_pos == i {
		return A[i]
	} else if piv_pos > i {
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
	random := start + rand.Int()%(end-start+1)>>1
	A[start], A[random] = A[random], A[start]
	return partition(A, start, end)
}
```


复杂度分析

- 时间复杂度：O(n)，如上文所述，证明过程可以参考「《算法导论》9.2：期望为线性的选择算法」。
- 空间复杂度：O(logn)，递归使用栈空间的空间代价的期望为 O(logn)。



## 方法二：基于堆排序的选择方法

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
	if lson < heap_size && A[largest] < A[lson] {
		largest = lson
	}
	if rson < heap_size && A[largest] < A[rson] {
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



[补充题6. 手撕堆排序 912. 排序数组](https://leetcode-cn.com/problems/sort-an-array/)

### 方法一：堆排序（大根堆）
思路和算法

堆排序的思想就是先将待排序的序列建成大根堆，使得每个父节点的元素大于等于它的子节点。此时整个序列最大值即为堆顶元素，我们将其与末尾元素交换，使末尾元素为最大值，然后再调整堆顶元素使得剩下的 n−1 个元素仍为大根堆，再重复执行以上操作我们即能得到一个有序的序列。

```go
func sortArray(A []int) []int {
	heap_sort(A)
	return A
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
	if lson < heap_size && A[largest] < A[lson] {
		largest = lson
	}
	if rson < heap_size && A[largest] < A[rson] {
		largest = rson
	}
	if i != largest {
		A[i], A[largest] = A[largest], A[i]
		max_heapify(A, largest, heap_size)
	}
}
```


复杂度分析

- 时间复杂度：O(nlogn)。初始化建堆的时间复杂度为 O(n)，建完堆以后需要进行 n−1 次调整，一次调整（即 maxHeapify） 的时间复杂度为 O(logn)，那么 n−1 次调整即需要 O(nlogn) 的时间复杂度。因此，总时间复杂度为 O(n+nlogn)=O(nlogn)。

- 空间复杂度：O(1)。只需要常数的空间存放若干变量。






[补充题5. 手撕归并排序 912. 排序数组](https://leetcode-cn.com/problems/sort-an-array/)


思路

归并排序利用了分治的思想来对序列进行排序。
对一个长为 n 的待排序的序列，我们将其分解成两个长度为 n/2 的子序列。
每次先递归调用函数使两个子序列有序，然后我们再线性合并两个有序的子序列使整个序列有序。

1. 确定分解点: mid := (l + r) / 2
2. 递归排序左右两边
3. 归并


```go

func sortArray(A []int) []int {
	merge_sort(A, 0, len(A)-1)
	return A
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



复杂度分析

- 时间复杂度：O(nlogn)。由于归并排序每次都将当前待排序的序列折半成两个子序列递归调用，然后再合并两个有序的子序列，而每次合并两个有序的子序列需要 O(n) 的时间复杂度，所以我们可以列出归并排序运行时间 T(n) 的递归表达式：

T(n)=2T(n/2)+O(n)

​ 根据主定理我们可以得出归并排序的时间复杂度为 O(nlogn)。

- 空间复杂度：O(n)。我们需要额外 O(n) 空间的 tmp 数组，且归并排序递归调用的层数最深为 log_2 n，所以我们还需要额外的 O(logn) 的栈空间，所需的空间复杂度即为 O(n+logn)=O(n)。








[785. 快速排序](https://www.acwing.com/problem/content/description/787/)

```go
package main 

import "fmt"

func quickSort(q []int, l, r int) {
    if l >= r { // 终止条件
        return
    }
    x := q[(l+r)>>1] // 确定分界点
    i, j := l-1, r+1  // 两个指针，因为do while要先自增/自减
    for i < j {       // 每次迭代
        for i++; q[i] < x; i++{}// do while 语法:交换后指针要移动，避免没必要的交换
        for j--; q[j] > x; j--{}
        if i < j { // swap 两个元素
            q[i], q[j] = q[j], q[i]
        }
    }
    quickSort(q, l, j) // 递归处理左右两段
    quickSort(q, j+1, r)
}

func main() {
    var n int 
    fmt.Scanf("%d", &n)
    q := make([]int, n)
    for i := 0; i < n; i ++ {
        fmt.Scanf("%d", &q[i])
    }
    quickSort(q, 0, n-1)
    for i := 0; i < n; i ++ {
        fmt.Printf("%d ",q[i] )
    }
}
```



```go
package main 

import "fmt"

func quickSort(q []int, l, r int) {
    if l >= r { // 终止条件
        return
    }
    x := q[(l+r)>>1] // 确定分界点
    i, j := l-1, r+1  // 两个指针，因为do while要先自增/自减
    for i < j {       // 每次迭代
        for { // do while 语法
            i++ // 交换后指针要移动，避免没必要的交换
            if q[i] >= x {
                break
            }
        }
        for {
            j--
            if q[j] <= x {
                break
            }
        }
        if i < j { // swap 两个元素
            q[i], q[j] = q[j], q[i]
        }
    }
    quickSort(q, l, j) // 递归处理左右两段
    quickSort(q, j+1, r)
}

func main() {
    var n int 
    fmt.Scanf("%d", &n)
    q := make([]int, n)
    for i := 0; i < n; i ++ {
        fmt.Scanf("%d", &q[i])
    }
    quickSort(q, 0, n-1)
    for i := 0; i < n; i ++ {
        fmt.Printf("%d ",q[i] )
    }
}


```







## 下面记录可忽略
------



![常见的排序算法的时间复杂度.png](http://ww1.sinaimg.cn/large/007daNw2ly1go45x0kga6j32a017e4cd.jpg)

1. 快速排序

```go
func sortArray(A []int) []int {
	quickSort(A, 0, len(A)-1)
	return A
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
func sortArray(A []int) []int {
    rand.Seed(time.Now().UnixNano())
	quickSort(A, 0, len(A)-1)
	return A
}
func quickSort(a []int, l, r int) {
	if l < r {
		pos := randomPartition(a, l, r)
		quickSort(a, l, pos-1)
		quickSort(a, pos+1, r)
	}
}
func randomPartition(a []int, l, r int) int {
	i := rand.Int() % (r - l + 1) + l 
	a[i], a[r] = a[r], a[i]
	return partition(a, l, r)
}
func partition(a []int, l, r int) int {
	x, i := a[r], l - 1
	for j := l; j < r; j ++ {
		if a[j] <= x {
			i ++ 
			a[i], a[j] = a[j], a[i]
		}
	}
	a[i+1], a[r] = a[r], a[i+1]
	return i+1
}

```



```go
func sortArray(A []int) []int {
    quickSort(A, 0, len(A)-1)
    return A
}
func quickSort (A[]int, l, r int) {
    if l >= r {
        return
    }
    p := A[(l+r)>>1]
    i, j := l-1, r+1
    for i < j {
        for {
            i ++ 
            if A[i] >= p {
                break
            }
        }
        for {
            j --
            if A[j] <= p {
                break
            }
        }
        if i < j {
            A[i], A[j] = A[j], A[i]
        }
    }
    quickSort(A, l, j)
    quickSort(A, j+1, r)
}
```


2. 堆排序

```go
func heapSort(a []int) {
    heapSize := len(a) 
    buildMaxHeap(a, heapSize)
    for i := heapSize - 1; i >= 1; i -- {
        a[0], a[i] = a[i], a[0]
        heapSize -- 
        maxHeapify(a, 0, heapSize)
    }
}

func buildMaxHeap(a[]int, heapSize int) {
    for i := heapSize / 2; i >= 0; i -- {
        maxHeapify(a, i, heapSize)
    }
} 

func maxHeapify(a []int, i, heapSize int) {
    l, r, largest := i * 2 + 1, i * 2 + 2, i 
    if l < heapSize && a[largest] < a[l] {
        largest = l 
    } 
    if r < heapSize && a[largest] < a[r] {
        largest = r 
    }
    if largest != i {
        a[largest], a[i] = a[i], a[largest]
        maxHeapify(a, largest, heapSize)
    }
}
```

3. 归并排序

```go
func mergeSort(a []int, l, r int) {
	if l >= r {
		return
	}
	mid := (l + r) / 2 
	mergeSort(a, l, mid)
	mergeSort(a, mid + 1, r)
    tmp := []int {}
	i, j := l, mid + 1
	for i <= mid || j <= r {
		if i > mid || (j <= r && a[j] < a[i]) {
			tmp = append(tmp, a[j])
			j++
		} else {
			tmp = append(tmp, a[i])
			i++
		}
	}
	copy(a[l: r + 1], tmp)
}
```

4. 选择排序

```go
func selectSort(a []int) {
	for i := 0; i < len(a)-1; i++ {
		minIdx := i
		for j := i + 1; j < len(a); j++ {
			if a[j] < a[minIdx] {
				minIdx = j
			}
		}
		a[i], a[minIdx] = a[minIdx], a[i]
	}
}
```

5. 插入排序

``` go
func insertionSort(A []int) {
	for i := 1; i < len(A); i++ {
		tmp := A[i]
		j := i - 1
		for j >= 0 && A[j] > tmp {
			A[j+1] = A[j] //向后移动1位
			j--                 //向前扫描
		}
		A[j+1] = tmp //添加到小于它的数的右边
	}
}
```

6. 冒泡排序

```Golang
func bubble_sort(A []int) {
	for i := 0; i < len(A); i++ {
		for j := 0; j < len(A)-i-1; j++ { //最后剩一个数不需比较-1
			if A[j] > A[j+1] {
				A[j], A[j+1] = A[j+1], A[j]
			}
		}
	}
}
```

7. 计数排序

```Golang
func count_sort(A []int) {
	cnt := [100001]int{}
	for i := 0; i < len(A); i++ {
		cnt[A[i]+50000] ++ //防止负数导致数组越界
	}
	for i, idx := 0, 0; i < 100001; i++ {
		for cnt[i] > 0 {
			A[idx] = i - 50000
			idx++
			cnt[i] --
		}
	}
}
```

8. 桶排序

``` go

```

9. 基数排序

``` go

```





















------


```go
func sortArray(A []int) []int {
    heapSort(A)
    return A
}

func heapSort(a []int) {
    heapSize := len(a) - 1
    buildMaxHeap(a, heapSize)
    for i := heapSize; i >= 1; i -- {
        a[0], a[i] = a[i], a[0]
        heapSize -- 
        maxHeapify(a, 0, heapSize)
    }
}

func buildMaxHeap(a[]int, heapSize int) {
    for i := heapSize / 2; i >= 0; i -- {
        maxHeapify(a, i, heapSize)
    }
} 

func maxHeapify(a []int, i, heapSize int) {
    l, r, largest := i * 2 + 1, i * 2 + 2, i 
    if l <= heapSize && a[largest] < a[l] {
        largest = l 
    } 
    if r <= heapSize && a[largest] < a[r] {
        largest = r 
    }
    if largest != i {
        a[largest], a[i] = a[i], a[largest]
        maxHeapify(a, largest, heapSize)
    }
}
```

```Golang
func quick_sort(A []int, l, r int) {
	if l >= r {
		return
	}
	A[r], A[(l+r)>>1] = A[(l+r)>>1], A[r]
	i := l - 1
	for j := l; j < r; j++ {
		if A[j] < A[r] {
			i++
			A[i], A[j] = A[j], A[i]
		}
	}
	i++
	A[i], A[r] = A[r], A[i]
	quick_sort(A, l, i-1)
	quick_sort(A, i+1, r)
}
```

```Golang
func merge_sort(A []int, l, r int) {
	if l >= r {
		return
	}
	mid := (l + r) >> 1
	merge_sort(A, l, mid)
	merge_sort(A, mid+1, r)
	i, j := l, mid+1
	tmp := []int{}
	for i <= mid || j <= r {
		if i > mid || (j <= r && A[j] < A[i]) {
			tmp = append(tmp, A[j])
			j++
		} else {
			tmp = append(tmp, A[i])
			i++
		}
	}
	copy(A[l:r+1], tmp)
}
```

```go
func sortArray(A []int) []int {
	n := len(A)
	temp := make([]int, n)
	mergeSort(A, temp, 0, n-1)
	return A
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

```Golang
func heap_sort(A []int) {
	lens := len(A) - 1
	for i := lens << 1; i >= 0; i-- {//建堆O(n)
		down(A, i, lens)
	}
	for j := lens; j >= 1; j-- {
		A[0], A[j] = A[j], A[0]
		lens--
		down(A, 0, lens)
	}
}
func down(A []int, i, lens int) {//O(logn)
	max := i
	if i<<1+1 <= lens && A[i<<1+1] > A[max] {
		max = i<<1 + 1
	}
	if i<<1+2 <= lens && A[i<<1+2] > A[max] {
		max = i<<1 + 2
	}
	if i != max {
		A[i], A[max] = A[max], A[i]
		down(A, max, lens)
	}
}
```

```Golang
func select_sort(A []int) {
	for i := 0; i < len(A)-1; i++ {
		pos := i
		for j := i + 1; j < len(A); j++ {
			if A[j] < A[pos] {
				pos = j
			}
		}
		A[i], A[pos] = A[pos], A[i]
	}
}
```

```Golang
func insert_sort(A []int) {
	for i := 1; i < len(A); i++ {
		tmp := A[i]
		j := i - 1
		for j >= 0 && A[j] > tmp {
			A[j+1] = A[j] //向后移动1位
			j--                 //向前扫描
		}
		A[j+1] = tmp //添加到小于它的数的右边
	}
}
```


```Golang
func bubble_sort(A []int) {
	for i := 0; i < len(A); i++ {
		for j := 0; j < len(A)-i-1; j++ { //最后剩一个数不需比较-1
			if A[j] > A[j+1] {
				A[j], A[j+1] = A[j+1], A[j]
			}
		}
	}
}
```


```Golang
func count_sort(A []int) {
	cnt := [100001]int{}
	for i := 0; i < len(A); i++ {
		cnt[A[i]+50000] ++ //防止负数导致数组越界
	}
	for i, idx := 0, 0; i < 100001; i++ {
		for cnt[i] > 0 {
			A[idx] = i - 50000
			idx++
			cnt[i] --
		}
	}
}
```



------

[912. 排序数组](https://leetcode-cn.com/problems/sort-an-array/)

* 考点1：能否实现解法的优化
* 考点2：是否了解快速选择算法
* 考点3：能否说明堆算法和快速选择算法的适用场景

## 方法一：快速排序

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
func sortArray(A []int) []int {
    rand.Seed(time.Now().UnixNano())
    quickSort(A, 0, len(A)-1)
    return A
}
func quickSort(a []int, l, r int) {
    if l >= r {
        return
    }
    pos := partition(a, l, r)
    quickSort(a, l, pos - 1)
    quickSort(a, pos + 1, r)
}
func partition(a []int, l, r int) int {
    p := rand.Int()%(r - l + 1) + l 
    a[r],a[p] = a[p], a[r] 
    i := l - 1
    for j := l; j < r; j ++ {
        if a[j] <= a[r] {
            i ++ 
            a[i], a[j] = a[j], a[i]
        }
    }
    a[i+1], a[r] = a[r], a[i+1]
    return i + 1
}
```

## 方法二：堆排序

思路和算法

堆排序的思想就是先将待排序的序列建成大根堆，使得每个父节点的元素大于等于它的子节点。此时整个序列最大值即为堆顶元素，
我们将其与末尾元素交换，使末尾元素为最大值，然后再调整堆顶元素使得剩下的 n−1 个元素仍为大根堆，再重复执行以上操作我们即能得到一个有序的序列。


```go

// 在大根堆中、最大元素总在根上，堆排序使用堆的这个属性进行排序
func HeapSort(A []int) {
	heap_size := len(A)
	build_maxheap(A, heap_size)
	for i := heap_size - 1; i >= 0; i-- {
		A[0], A[i] = A[i], A[0]      // 交换堆顶元素 A[0] 与堆底元素 A[i]，最大值 A[0] 放置在数组末尾
		heap_size--                  // 删除堆顶元素 A[0]
		max_heapify(A, 0, heap_size) // 堆顶元素 A[0] 向下调整
	}
}
func build_maxheap(A []int, heap_size int) { // 建堆 O(n)
	for i := heap_size / 2; i >= 0; i-- { // heap_size>>1 后面都是叶子节点，不需要向下调整
		max_heapify(A, i, heap_size)
	}
}
func max_heapify(A []int, i, heap_size int) { // 调整大根堆 O(nlogn)
	lson, rson, largest := i*2+1, i*2+2, i        // i<<1+1, i<<1+2
	if lson < heap_size && A[largest] < A[lson] { // 左儿子存在并大于根
		largest = lson
	}
	if rson < heap_size && A[largest] < A[rson] { // 右儿子存在并大于根
		largest = rson
	}
	if i != largest { // 找到左右儿子的最大值
		A[i], A[largest] = A[largest], A[i] // 堆顶调整为最大值
		max_heapify(A, largest, heap_size)  // 递归调整子树
	}
}

// 迭代
func maxHeapify(A []int, i, heapSize int) {
	for i<<1+1 < heapSize {
		lson, rson, large := i<<1+1, i<<1+2, i
		if lson < heapSize && A[large] < A[lson] {
			large = lson
		}
		for rson < heapSize && A[large] < A[rson] {
			large = rson
		}
		if large != i {
			A[i], A[large] = A[large], A[i]
			i = large
		} else {
			break
		}
	}
}
```

## 方法三：归并排序
思路

归并排序利用了分治的思想来对序列进行排序。
对一个长为 n 的待排序的序列，我们将其分解成两个长度为 n/2 的子序列。
每次先递归调用函数使两个子序列有序，然后我们再线性合并两个有序的子序列使整个序列有序。

1. 确定分解点: mid := (l + r) / 2
2. 递归排序左右两边
3. 归并

```go
func sortArray(A []int) []int {
	mergeSort(A, 0, len(A)-1)
	return A
}
func mergeSort(a []int, l, r int) {
	if l >= r {
		return
	}
	mid := (l + r) / 2
	mergeSort(a, l, mid)
	mergeSort(a, mid+1, r)
	tmp := []int{}
	i, j := l, mid+1
	for i <= mid || j <= r {
		if i > mid || (j <= r && a[j] < a[i]) {
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

## 方法4：选择排序

从未排序序列中找到最小元素，存放到排序序列的起始位置
再从剩余元素中找到最小元素，存放到已排序序列末尾...


```go
func sortArray(A []int) []int {
	selectSort(A)
	return A
}

func selectSort(a []int) {
	for i := 0; i < len(a)-1; i++ {
		minIdx := i
		for j := i + 1; j < len(a); j++ {
			if a[j] < a[minIdx] {
				minIdx = j
			}
		}
		a[i], a[minIdx] = a[minIdx], a[i]
	}
}
```
Time Limit Exceeded
11/11 cases passed (N/A)

# 5.插入排序

每次将一个待排序的序列插入到一个前面已排好序的子序列当中

构建有序序列，对于未排序数据，在已排序序列中从后向前扫描，找到相应位置并插入。
选取数组第2个元素开始比较，如果左边第1个元素比它大，左边元素向右移动1位，索引减1，向前扫描...
直到左边元素比他小，插入这个元素右边

```go
func insertionSort(A []int) {
	for i := 1; i < len(A); i++ {
		tmp := A[i]
		j := i - 1
		for j >= 0 && A[j] > tmp {
			A[j+1] = A[j] //向后移动1位
			j--                 //向前扫描
		}
		A[j+1] = tmp //添加到小于它的数的右边
	}
}
```














------



# 1. 快速排序

### 模板一：

快排思路：
1. 确定分界点 x：q[l], q[r], q[(l+r)>>1], 随机
2. 调整区间：left <= x, right >= x
3. 递归处理左右两边

时间复杂度： O(nlog(n)) 
空间复杂度： O(log(n)), 递归使用栈空间的空间代价为O(logn)。

C++ 模板一：

```c++ 
void quickSort(int a[], int l, int r) {
	if (l >= r) return;
	int i = l-1, j =r+1, x = a[l+r>>1];
	while(i < j) {
		do i++; while a[i] < x;
		do j--; while a[j] > x;
		if(i<j) swap(a[i],a[j]);
	}
	quickSort(a, l, j);
	quickSort(a, j+1, r);
}
```
C++ 模板二：

```c++
void quickSort(int a[], int l, int r) {
	if (l >= r) return;
	int i = l-1, j =r+1, x = a[l+r>>1];
	while(i < j) {
		while a[++ i] < x;
		while a[-- j] > x;
		if(i<j) swap(a[i],a[j]);
	}
	quickSort(a, l, j);
	quickSort(a, j+1, r);
}
```

 go 模板一：

```go
func sortArray(A []int) []int {
    quickSort(A, 0, len(A)-1)
    return A
}
func quickSort (q[]int, l, r int) {
    if l >= r {
        return
    }
    p := q[(l+r)>>1]
    i, j := l-1, r+1
    for i < j {
        for {
            i ++ 
            if q[i] >= p {
                break
            }
        }
        for {
            j -- 
            if q[j] <= p {
                break
            }
        }
        if i < j {
            q[i], q[j] = q[j], q[i]
        }
    }
    quickSort(q, l, j)
    quickSort(q, j+1, r)
}
```

 go 模板二：

```go 
func sortArray(A []int) []int {
    quickSort(A, 0, len(A)-1)
    return A
}
func quickSort (q[]int, l, r int) {
    if l >= r {
        return
    }
    p := q[(l+r)>>1]
    i, j := l-1, r+1
    for i < j {
        for {
            i ++ 
            if q[i] >= p {
                break
            }
        }
        for {
            j -- 
            if q[j] <= p {
                break
            }
        }
        if i < j {
            q[i], q[j] = q[j], q[i]
        }
    }
    quickSort(q, l, j)
    quickSort(q, j+1, r)
}
```
时间复杂度： O(n)
n + n/2 + n/4 + ... = (1 + 1/2 + 1/4 + ...)n <= 2n 

## 方法二

```go
func sortArray(A []int) []int {
    rand.Seed(time.Now().UnixNano())
	quickSort(A, 0, len(A)-1)
	return A
}
func quickSort(a []int, l, r int) {
	if l < r {
		pos := randomPartition(a, l, r)
		quickSort(a, l, pos-1)
		quickSort(a, pos+1, r)
	}
}
func randomPartition(a []int, l, r int) int {
	i := rand.Int() % (r - l + 1) + l 
	a[i], a[r] = a[r], a[i]
	return partition(a, l, r)
}
func partition(a []int, l, r int) int {
	x, i := a[r], l - 1
	for j := l; j < r; j ++ {
		if a[j] <= x {
			i ++ 
			a[i], a[j] = a[j], a[i]
		}
	}
	a[i+1], a[r] = a[r], a[i+1]
	return i+1
}
```

## 方法三： 


```go 
func sortArray(A []int) []int {
    quickSort(A, 0, len(A)-1)
    return A
}
func quickSort(a []int, l, r int) {
    if l >= r {
        return
    }
    a[r], a[(l+r)>>1] = a[(l+r)>>1], a[r]
    i := l - 1
    for j := l; j < r; j ++ {
        if a[j] < a[r] {
            i ++
            a[i], a[j] = a[j], a[i]
        }
    }
    i ++
    a[i], a[r] = a[r], a[i]
    quickSort(a, l, i-1 )
    quickSort(a, i+1, r)
}
```

```go 
func sortArray(A []int) []int {
    quickSort(A, 0, len(A)-1)
    return A
}
func quickSort(A []int, l, r int) {
    if l >= r {
        return
    }
    A[r], A[(l+r)>>1] = A[(l+r)>>1], A[r]
    i := l - 1
    for j := l; j < r; j++ {
        if A[j] < A[r] {
            i++
            A[i], A[j] = A[j], A[i]
        }
    }
    i ++
    A[i], A[r] = A[r], A[i]
    quickSort(A, l, i-1)
    quickSort(A, i+1, r)
}
```



```go
func quick_sort(A []int, start, end int) {
	if start >= end {
		return
	}
	piv := A[start+(end-start)>>1]
	i, j := start-1, end+1
	for i < j {
		for {
			i++
			if A[i] >= piv {
				break
			}
		}
		for {
			j--
			if A[j] <= piv {
				break
			}
		}
		if i < j {
			A[i], A[j] = A[j], A[i]
		}
	}
	quick_sort(A, start, j)
	quick_sort(A, j+1, end)
}
```


```go
func sortArray(A []int) []int {
	n := len(A)
	temp := make([]int, n)
	mergeSort(A, temp, 0, n-1)
	return A
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

### 方法二：归并排序



```go

func sortArray(A []int) []int {
	n := len(A)
	mergeSort(A, 0, n-1)
	return A
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
func sortArray(A []int) []int {
	mergeSort(A, 0, len(A)-1)
	return A
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
func sortArray(A []int) []int {
	mergeSort(A, 0, len(A)-1)
	return A
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