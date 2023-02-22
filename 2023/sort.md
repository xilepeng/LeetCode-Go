
1. [补充题4. 手撕快速排序 ](#补充题4-手撕快速排序-)
2. [补充题5. 手撕归并排序](#补充题5-手撕归并排序)
3. [补充题6. 手撕堆排序 ](#补充题6-手撕堆排序-)
4. [插入、选择、冒泡排序](#插入选择冒泡排序)
5. [排序总结](#排序总结)



## [补充题4. 手撕快速排序 ](https://leetcode-cn.com/problems/sort-an-array/)


```go

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
```


## [补充题5. 手撕归并排序](https://leetcode.cn/problems/sort-an-array/)

```go
func sortArray(nums []int) []int {
	MergeSort(nums, 0, len(nums)-1)
	return nums
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
```

## [补充题6. 手撕堆排序 ](https://leetcode-cn.com/problems/sort-an-array/)

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
func maxHeapify(nums []int, i, heapSize int) {
	for i<<1+1 < heapSize {
		lson, rson, large := i<<1+1, i<<1+2, i
		if lson < heapSize && nums[large] < nums[lson] {
			large = lson
		}
		for rson < heapSize && nums[large] < nums[rson] {
			large = rson
		}
		if large != i {
			nums[i], nums[large] = nums[large], nums[i]
			i = large
		} else {
			break
		}
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
func sortArray(nums []int) []int {
	InsertSort(nums, len(nums))
	SelectionSort(nums, len(nums))
	BubbleSort(nums, len(nums))
	
	return nums
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

## 排序总结 
---

```go
/*
 * @lc app=leetcode.cn id=912 lang=golang
 *
 * [912] 排序数组
 */

// @lc code=start
func sortArray(nums []int) []int {
	// InsertSort(nums, len(nums))
	// SelectionSort(nums, len(nums))
	// BubbleSort(nums, len(nums))

	rand.Seed(time.Now().UnixNano())
	// QuickSort(nums, 0, len(nums)-1)
	// QuickSort1(nums, 0, len(nums)-1)

	MergeSort(nums, 0, len(nums)-1)
	// HeapSort(nums)

	return nums
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
func maxHeapify(nums []int, i, heapSize int) {
	for i<<1+1 < heapSize {
		lson, rson, large := i<<1+1, i<<1+2, i
		if lson < heapSize && nums[large] < nums[lson] {
			large = lson
		}
		for rson < heapSize && nums[large] < nums[rson] {
			large = rson
		}
		if large != i {
			nums[i], nums[large] = nums[large], nums[i]
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

// @lc code=end


```

